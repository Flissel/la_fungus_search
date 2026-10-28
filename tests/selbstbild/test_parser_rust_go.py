import dataclasses
import json

from selbstbild.parser.basis import parsen
from selbstbild.parser.rust_go import parse_go, parse_rust
from selbstbild.typen import Einstieg, Import

RUST = b'''use crate::kern::Lauf;
use a::{b, c};
use d::e as x;

impl Lauf {
    fn start(&self) { helfer(); x(); }
}

impl Tr for Lauf {
    fn lauf(&self) { std::env::var_os("ZWEI"); }
}

fn main() {
    std::env::var("OPENFANG_HOME");
    env::var("DREI");
    let _ = env!("VIER");
    let _ = option_env!("FUENF");
    let geheim = "wert-nicht-speichern";
}

#[test]
fn t() {}

#[tokio::test]
async fn t2() {}
'''

GO = b'''package main

import "os"
import (
	"net/http"
	f "fmt"
)

func (s *Server) Start() { run(); f.Println("x") }

func main() { os.Getenv("PORT"); os.LookupEnv("HOME2") }

func helfer() {}
'''


def test_rust():
    e = parse_rust("src/lib.rs", RUST)
    assert {"Lauf.start", "Lauf.lauf", "main", "t", "t2"} <= {s.qualname for s in e.symbole}
    assert Import(modul="crate::kern", namen=("Lauf",), relativ=False, zeile=1) in e.importe
    assert Import(modul="a", namen=("b", "c"), relativ=False, zeile=2) in e.importe
    assert Import(modul="d", namen=("e",), relativ=False, zeile=3) in e.importe
    assert {a.name for a in e.aufrufe if a.in_symbol == "Lauf.start"} == {"helfer", "e"}
    assert {v.name for v in e.env} == {"OPENFANG_HOME", "ZWEI", "DREI", "VIER", "FUENF"}
    assert {(x.art, x.symbol) for x in e.einstiege} == {
        ("main", "main"), ("test", "t"), ("test", "t2")}
    assert "wert-nicht-speichern" not in json.dumps(dataclasses.asdict(e))
    assert not e.fehlerhaft


def test_go():
    e = parse_go("cmd/x.go", GO)
    assert {"Server.Start", "main", "helfer"} <= {s.qualname for s in e.symbole}
    assert Import(modul="os", namen=(), relativ=False, zeile=3) in e.importe
    assert {i.modul for i in e.importe} == {"os", "net/http", "fmt"}
    assert {a.name for a in e.aufrufe if a.in_symbol == "Server.Start"} == {"run", "fmt.Println"}
    assert {v.name for v in e.env} == {"PORT", "HOME2"}
    assert [(x.art, x.symbol) for x in e.einstiege] == [("main", "main")]


def test_go_test_und_fremdes_paket():
    quelle = b"package x\nfunc TestX(t *testing.T) {}\nfunc BenchmarkY(b *testing.B) {}\nfunc main() {}\n"
    e = parse_go("a/x_test.go", quelle)
    assert {(x.art, x.symbol) for x in e.einstiege} == {
        ("test", "TestX"), ("test", "BenchmarkY")}
    e2 = parse_go("a/x.go", quelle)
    assert e2.einstiege == ()


def test_fehlerhaft_und_register():
    assert parse_rust("a.rs", b"fn ok() {} fn kaputt( {").fehlerhaft
    e = parse_go("a.go", b"package main\nfunc ok() {}\nfunc kaputt( {")
    assert e.fehlerhaft and "ok" in {s.qualname for s in e.symbole}
    assert parsen("rust", "a.rs", b"fn m() {}").symbole
    assert parsen("go", "a.go", b"package p\nfunc F() {}").symbole


def test_rust_tokio_mit_argumenten_und_env_import():
    q = (b"use std::env::var;" + chr(10).encode()
         + b'#[tokio::test(flavor = "multi_thread")]' + chr(10).encode()
         + b'async fn t3() { var("KEY_A"); }' + chr(10).encode())
    e = parse_rust("a.rs", q)
    assert ("test", "t3") in {(x.art, x.symbol) for x in e.einstiege}
    assert {v.name for v in e.env} == {"KEY_A"}


def test_go_kleinbuchstabe_nach_praefix_kein_test():
    q = b"package x" + chr(10).encode() + b"func Testing() {}" + chr(10).encode()         + b"func Test() {}" + chr(10).encode() + b"func TestA(t *testing.T) {}" + chr(10).encode()
    e = parse_go("a/x_test.go", q)
    assert {x.symbol for x in e.einstiege} == {"Test", "TestA"}
