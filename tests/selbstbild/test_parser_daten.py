import dataclasses
import json

from selbstbild.parser.basis import parsen
from selbstbild.parser.daten import (
    parse_bash, parse_dotenv, parse_json, parse_powershell, parse_sql, parse_yaml,
)
from selbstbild.typen import Einstieg

SQL = b'''-- Kommentar; mit Semikolon
CREATE TABLE IF NOT EXISTS marketing.kampagnen (
    id int primary key,
    name text DEFAULT 'a;b'
);
CREATE OR REPLACE FUNCTION public.zaehlen() RETURNS int AS $$
BEGIN
  PERFORM 1;
  RETURN 1;
END;
$$ LANGUAGE plpgsql;
/* CREATE TABLE nicht_da (x int); */
CREATE VIEW v AS SELECT 1;
'''


def test_sql_objekte():
    e = parse_sql("m/001.sql", SQL)
    assert {(o.art, o.name) for o in e.sql} == {
        ("tabelle", "marketing.kampagnen"),
        ("sql_funktion", "public.zaehlen"),
        ("view", "public.v"),
    }
    assert len(e.sql) == 3
    zeilen = {o.name: o.zeile for o in e.sql}
    assert zeilen["marketing.kampagnen"] == 2
    assert all(len(o.digest) == 64 for o in e.sql)


def test_sql_dollar_tag_und_crlf_digest():
    q = b'CREATE FUNCTION f() RETURNS int AS $body$ select 1; select 2; $body$ LANGUAGE sql;\nCREATE TABLE "Public"."T" (x int);\n'
    e = parse_sql("a.sql", q)
    assert {o.name for o in e.sql} == {"public.f", "public.t"}
    a = parse_sql("a.sql", b"CREATE TABLE t (x int);\n")
    b = parse_sql("a.sql", b"CREATE TABLE t (x int);\r\n")
    assert a.sql[0].digest == b.sql[0].digest


COMPOSE = b'''services:
  api:
    image: x
    environment: [API_KEY=geheim123, "PORT=8080", PLAIN]
  db:
    environment:
      DB_URL: ${DATABASE_URL}
'''


def test_yaml_env_ohne_werte():
    e = parse_yaml("docker-compose.yml", COMPOSE)
    assert {v.name for v in e.env} == {"API_KEY", "PORT", "PLAIN", "DB_URL", "DATABASE_URL"}
    s = json.dumps(dataclasses.asdict(e))
    assert "geheim123" not in s and "8080" not in s
    assert e.fehlerhaft is False


def test_yaml_default_und_mehrere_dokumente():
    e = parse_yaml("a.yaml", b"a: ${X_A:-wert}\n---\nb: $X_B\nc: 5\n")
    assert {v.name for v in e.env} == {"X_A", "X_B"}
    assert "wert" not in json.dumps(dataclasses.asdict(e))


def test_yaml_fehlerhaft():
    e = parse_yaml("a.yml", b"a: [1, 2\nb: : :\n")
    assert e.fehlerhaft is True


def test_dotenv():
    e = parse_dotenv(".env.example", b"OPENAI_API_KEY=sk-abc\n# KOMMENTAR=1\nexport B_X = 2\n\n")
    assert {v.name for v in e.env} == {"OPENAI_API_KEY", "B_X"}
    assert "sk-abc" not in json.dumps(dataclasses.asdict(e))


def test_powershell():
    e = parse_powershell("s.ps1", b'$env:VIBEMIND_TOKEN = "xyz"; Write-Host $env:HOME_DIR\n')
    assert {v.name for v in e.env} == {"VIBEMIND_TOKEN", "HOME_DIR"}
    assert "xyz" not in json.dumps(dataclasses.asdict(e))


def test_bash():
    e = parse_bash("s.sh", b'echo "${API_BASE:-http://a}" $TOKEN $1 $@ $klein ${klein2}\n')
    assert {v.name for v in e.env} == {"API_BASE", "TOKEN"}
    assert "http://a" not in json.dumps(dataclasses.asdict(e))


def test_package_json():
    q = b'{"name": "p", "scripts": {"start": "node server.js"}, "bin": {"x": "cli.js"}}'
    e = parse_json("package.json", q)
    assert Einstieg(art="cli_skript", symbol="start", zeile=1) in e.einstiege
    assert Einstieg(art="cli_skript", symbol="x", zeile=1) in e.einstiege
    assert len(e.einstiege) == 2


def test_package_json_bin_string_und_andere_json():
    e = parse_json("sub/package.json", b'{"name": "tool", "bin": "c.js"}')
    assert [x.symbol for x in e.einstiege] == ["tool"]
    assert parse_json("tsconfig.json", b'{"a": 1}').einstiege == ()
    assert parse_json("package.json", b"{kaputt").fehlerhaft is True


def test_registry():
    assert parsen("sql", "a.sql", b"CREATE TABLE t (x int);").sql
    assert parsen("dotenv", ".env.x", b"A_B=1").env
    assert parsen("bash", "a.sh", b"echo $A_B").env
    assert parsen("powershell", "a.ps1", b"$env:A_B").env
    assert parsen("yaml", "a.yml", b"k: ${A_B}").env
    assert parsen("json", "package.json", b'{"scripts":{"a":"b"}}').einstiege
