import dataclasses
import json

from selbstbild.parser.basis import parsen
from selbstbild.parser.typescript import parse_ts
from selbstbild.typen import Einstieg, Import

SRC = b'''
import { a as b } from "./util";
import x from "lib";
import * as ns from "m";

export class K { m() { b(); this.n(); ns.f(); } }
export const f = async () => process.env.API_URL;
export function g() { fetch("/api/v1/x"); const o = process.env["OTHER_KEY"]; }
'''


def test_hauptquelle():
    e = parse_ts("src/k.ts", SRC, "typescript")
    assert {s.qualname for s in e.symbole} == {"K", "K.m", "f", "g"}
    arten = {s.qualname: s.art for s in e.symbole}
    assert arten == {"K": "klasse", "K.m": "methode", "f": "funktion", "g": "funktion"}
    assert Import(modul="./util", namen=("a",), relativ=True, zeile=2) in e.importe
    assert Import(modul="lib", namen=("x",), relativ=False, zeile=3) in e.importe
    namen = {a.name for a in e.aufrufe if a.in_symbol == "K.m"}
    assert {"a", "this.n", "m.f"} <= namen
    assert {(v.name, v.in_symbol) for v in e.env} == {
        ("API_URL", "f"), ("OTHER_KEY", "g")}
    assert "/api/v1/x" not in json.dumps(dataclasses.asdict(e))
    assert e.fehlerhaft is False


def test_dispatch():
    assert parsen("typescript", "a.ts", SRC) == parse_ts("a.ts", SRC, "typescript")
    assert parsen("javascript", "a.js", b"function h() {}").symbole[0].qualname == "h"
    assert parsen("tsx", "a.tsx", b"export function K() { return <div/>; }").symbole[0].qualname == "K"


def test_next_route():
    q = b"export async function GET() { return 1 }\nexport function helfer() {}"
    e = parse_ts("app/api/kunden/route.ts", q, "typescript")
    assert [x for x in e.einstiege if x.art == "route"] == [Einstieg("route", "GET", 1)]
    e2 = parse_ts("lib/route.ts", q, "typescript")
    assert not e2.einstiege


def test_express_route():
    e = parse_ts("srv.js", b'app.get("/x", handler);\nrouter.use(mw);', "javascript")
    assert [x.art for x in e.einstiege] == ["route"]
    assert "/x" not in json.dumps(dataclasses.asdict(e))


def test_express_route_braucht_handler():
    q = (b'cache.delete("/tmp");\n'
         b'app.get("/a", h);\n'
         b'app.post("/b", (req, res) => res.end());\n'
         b'router.use("/c", function (req, res) {});\n'
         b'app.get("/d");\n'
         b'app.get("/e", "text");\n'
         b'store.get("/f", 1);\n')
    e = parse_ts("srv.js", q, "javascript")
    assert [(x.art, x.zeile) for x in e.einstiege] == [("route", 2), ("route", 3), ("route", 4)]


def test_env_destrukturierung():
    q = b"const { API_URL, X: y, Z = 1, ...rest } = process.env;\nconst { W } = import.meta.env;\n"
    e = parse_ts("cfg.ts", q, "typescript")
    assert [(v.name, v.zeile) for v in e.env] == [("API_URL", 1), ("X", 1), ("Z", 1), ("W", 2)]
    assert parse_ts("o.ts", b"const { A } = andere;\n", "typescript").env == ()


def test_re_exporte_sind_importe():
    q = b'export { Knopf, B as C } from "./knopf";\nexport * from "./x";\nexport * as ns from "./y";\nexport { lokal };\n'
    e = parse_ts("ui/index.ts", q, "typescript")
    assert e.importe == (
        Import(modul="./knopf", namen=("Knopf", "B"), zeile=1, relativ=True),
        Import(modul="./x", namen=("*",), zeile=2, relativ=True),
        Import(modul="./y", namen=("*",), zeile=3, relativ=True),
    )


def test_ts_test():
    q = b'describe("x", () => { it("y", () => {}); });'
    e = parse_ts("a.test.ts", q, "typescript")
    assert e.einstiege and all(x == Einstieg("test", None, x.zeile) for x in e.einstiege)
    assert not parse_ts("a.ts", q, "typescript").einstiege


def test_tsx():
    e = parse_ts("k.tsx", b"export function Karte() { return <div/> }", "tsx")
    assert {s.qualname for s in e.symbole} == {"Karte"}
    assert e.fehlerhaft is False


def test_fehlerhaft():
    e = parse_ts("k.ts", b"export function ok() {}\nfunction kaputt( {", "typescript")
    assert e.fehlerhaft is True
    assert "ok" in {s.qualname for s in e.symbole}
