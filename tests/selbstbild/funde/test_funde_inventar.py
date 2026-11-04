from __future__ import annotations

import builtins
import io
import pathlib
from pathlib import Path

import pytest

from selbstbild.funde.inventar import (
    Inventar, darf_oeffnen, inventar, inventar_fund, inventar_tabelle,
)
from selbstbild.funde.text import kappen_zeile, pruefen
from selbstbild.typen import Beleg, Kante, Knoten

WERTE = ("geheim123", "wert456", "wert789", "wert000", "psgeheim", "echtgeheim", "lokalgeheim",
         "secretsgeheim", "PEMGEHEIM", "hostgeheim")
VERBOTEN = (".env", ".env.local", "secrets.env", "cert.pem")

LLM = """\
from anthropic import Anthropic
from openai import AsyncOpenAI as AO
import ollama
from vibemind_shared.llm import get_client


def bauen():
    a = Anthropic()
    b = AO(model="gpt-test", api_key="x")
    c = ollama.Client(host="h")
    d = get_client("captain")
    return a, b, c, d
"""

FREMD_GET_CLIENT = """\
from meinpaket import get_client

x = get_client("nicht-vibemind")
y = OpenAI(model=MODELL)
"""

COMPOSE = """\
services:
  app:
    image: x
    environment:
      FOO: ${FOO}
      SETONLY: wert789
      HOSTED: ${FROM_HOST:-hostgeheim}
  worker:
    environment:
      - LISTED=wert000
"""

DIENSTE = """\
dienste:
  alpha:
    port: 8000
    ort: pc
  beta:
    port: 8000
    ort: container
  gamma:
    port: [9000, 9001]
  delta:
    port: null
  epsilon:
    port: 7000
"""


def _kante(name: str, repo: str, pfad: str, zeile: int) -> Kante:
    return Kante("liest_env", f"datei:{repo}:{pfad}", f"env:{name}", True, Beleg(repo, pfad, zeile, name, "d"))


def _graph(graph_stand, *extra: Kante) -> Path:
    kanten = [
        _kante("FOO", "aussen", "app/main.py", 3),
        _kante("ONLY_READ", "aussen", "app/main.py", 5),
        _kante("BAR_KEY", "aussen", "app/main.py", 7),
        _kante("LISTED", "aussen", "app/worker.ts", 2),
        _kante("PATH", "aussen", "app/main.py", 9),
        # Stufe 1 vermischt Setzen und Lesen in diesen Dateiarten: sie werden aus dem Graphen nicht genommen
        _kante("NUR_IM_BEISPIEL_GRAPH", "aussen", ".env.example", 1),
        _kante("NUR_IN_COMPOSE_GRAPH", "aussen", "docker-compose.yml", 1),
        _kante("NUR_IN_PS_GRAPH", "aussen", "skripte/start.ps1", 1),
        _kante("NUR_IN_ENVLOCAL_GRAPH", "aussen", ".env.local", 1),
        *extra,
    ]
    knoten = [Knoten(k.nach, "env", k.nach[4:], None) for k in kanten]
    return graph_stand(knoten, kanten)


def _repo(git_repo) -> Path:
    r = git_repo
    r.datei("selbstbild.yaml", "name: aussen\ndienste: config/dienste.yaml\n")
    r.datei(".env.example", "# FOO=kommentar\nFOO=geheim123\nexport BAR_KEY = wert456\nBAD-NAME=x\n\nOHNE_GLEICH\n")
    r.datei(".env", "FOO=echtgeheim\n")
    r.datei(".env.local", "FOO=lokalgeheim\n")
    r.datei("secrets.env", "FOO=secretsgeheim\n")
    r.datei("cert.pem", "-----BEGIN PRIVATE KEY-----\nPEMGEHEIM\n")
    r.datei("docker-compose.yml", COMPOSE)
    r.datei("skripte/start.ps1", "# $env:KOMMENTIERT = 1\n$env:PS_SET = \"psgeheim\"\nWrite-Host $env:PS_READ\n"
                                 "${env:PS_KLAMMER} = 2\nif ($env:PS_CMP -eq 1) {}\n")
    r.datei("config/dienste.yaml", DIENSTE)
    r.datei("app/llm.py", LLM)
    r.datei("app/fremd.py", FREMD_GET_CLIENT)
    r.datei("app/kaputt.py", "def x(:\n    OpenAI(\n")
    r.commit("init")
    # nicht versioniert: wird nie gelesen (R-a)
    (r.wurzel / "node_modules" / "p").mkdir(parents=True)
    (r.wurzel / "node_modules" / "p" / ".env.example").write_text("UNVERSIONIERT=1\n", encoding="utf-8")
    return r.wurzel


def _eintrag(inv: Inventar, name: str):
    treffer = [n for n in inv.namen if n.name == name]
    assert len(treffer) == 1, (name, [n.name for n in inv.namen])
    return treffer[0]


def _alle_ausgaben(inv: Inventar) -> str:
    f = inventar_fund(inv)
    return "\n".join([repr(inv), f.titel, f.text, f.anker, repr(f.belege), inventar_tabelle(inv)])


def test_env_nur_namen(git_repo, graph_stand) -> None:
    wurzel = _repo(git_repo)
    inv = inventar(wurzel, stand_unter=_graph(graph_stand))
    ausgabe = _alle_ausgaben(inv)
    for wert in WERTE:
        assert wert not in ausgabe, wert
    foo = _eintrag(inv, "FOO")
    assert foo.art == "env"
    assert "aussen/.env.example:2" in foo.gesetzt
    assert any(o.startswith("aussen/docker-compose.yml:") for o in foo.gesetzt)
    bar = _eintrag(inv, "BAR_KEY")
    assert bar.art == "key" and bar.gesetzt == ["aussen/.env.example:3"]
    namen = {n.name for n in inv.namen}
    assert "BAD-NAME" not in namen and "OHNE_GLEICH" not in namen and "UNVERSIONIERT" not in namen
    # Compose: Liste links vom `=`; `${NAME}` im Wert ist eine Lesestelle, die Vorgabe danach nie
    assert _eintrag(inv, "LISTED").gesetzt == ["aussen/docker-compose.yml:10"]
    assert _eintrag(inv, "FROM_HOST").gelesen == ["aussen/docker-compose.yml:7"]


def test_echte_env_nie_gelesen(git_repo, graph_stand, monkeypatch: pytest.MonkeyPatch) -> None:
    wurzel = _repo(git_repo)
    stand = _graph(graph_stand)
    geoeffnet: list[str] = []

    def waechter(orig):
        def offen(datei, *a, **k):
            name = Path(str(datei)).name
            if name in VERBOTEN:
                geoeffnet.append(name)
                raise AssertionError(f"verbotene Datei geoeffnet: {name}")
            return orig(datei, *a, **k)
        return offen

    monkeypatch.setattr(builtins, "open", waechter(builtins.open))
    monkeypatch.setattr(io, "open", waechter(io.open))
    for meth in ("open", "read_text", "read_bytes"):
        orig = getattr(pathlib.Path, meth)

        def methode(self, *a, _orig=orig, **k):
            if self.name in VERBOTEN:
                geoeffnet.append(self.name)
                raise AssertionError(f"verbotene Datei geoeffnet: {self.name}")
            return _orig(self, *a, **k)
        monkeypatch.setattr(pathlib.Path, meth, methode)

    # Gegenprobe: der Waechter greift wirklich
    with pytest.raises(AssertionError):
        (wurzel / ".env").read_text()
    geoeffnet.clear()

    inv = inventar(wurzel, stand_unter=stand)
    assert geoeffnet == []
    assert _eintrag(inv, "FOO").gesetzt  # das Inventar lief durch
    for name in VERBOTEN:
        assert not darf_oeffnen(name)
    for name in (".env.example", ".env.sample", ".env.template", "dev.example.env", "x.sample.env",
                 "y.template.env", ".ENV.EXAMPLE"):
        assert darf_oeffnen(name), name
    for name in (".env.production", ".env.sanitized", "prod.env", "example.env", "key.PEM"):
        assert not darf_oeffnen(name), name


def test_port_widerspruch(git_repo, graph_stand) -> None:
    wurzel = _repo(git_repo)
    inv = inventar(wurzel, stand_unter=_graph(graph_stand))
    p8000 = _eintrag(inv, "8000")
    assert p8000.art == "port"
    assert p8000.gesetzt == ["aussen/config/dienste.yaml:3 (alpha)", "aussen/config/dienste.yaml:6 (beta)"]
    assert p8000.befund == ["widerspruechlich: Port 8000 für alpha, beta"]
    assert _eintrag(inv, "9000").befund == ["widerspruechlich: Dienst gamma mit Ports 9000, 9001"]
    assert _eintrag(inv, "9001").befund == ["widerspruechlich: Dienst gamma mit Ports 9000, 9001"]
    assert _eintrag(inv, "7000").befund == []
    assert not any(n.name in ("null", "None") for n in inv.namen if n.art == "port")


def test_llm_clients_erkannt(git_repo, graph_stand) -> None:
    wurzel = _repo(git_repo)
    inv = inventar(wurzel, stand_unter=_graph(graph_stand))
    fundstellen = {(c.anbieter, c.modell, c.fundstelle) for c in inv.clients}
    assert fundstellen == {
        ("anthropic", None, "aussen/app/llm.py:8"),
        ("openai", "gpt-test", "aussen/app/llm.py:9"),
        ("ollama", None, "aussen/app/llm.py:10"),
        ("vibemind_shared", None, "aussen/app/llm.py:11"),
        ("openai", None, "aussen/app/fremd.py:4"),  # Modell kein Literal -> keins
    }
    assert _eintrag(inv, "openai").art == "llm_client"
    assert _eintrag(inv, "openai").gesetzt == ["aussen/app/fremd.py:4", "aussen/app/llm.py:9 model=gpt-test"]
    assert inv.uebersprungen == 1  # app/kaputt.py, gezaehlt statt Absturz
    assert any("1 .py-Datei" in n for n in inv.notizen)
    text = inventar_fund(inv).text
    assert "4 LLM-Clients ohne `get_client(role)`: auf `vibemind_shared.get_client` umstellen" in text


def test_tot_und_ungesetzt(git_repo, graph_stand) -> None:
    wurzel = _repo(git_repo)
    inv = inventar(wurzel, stand_unter=_graph(graph_stand))
    assert _eintrag(inv, "SETONLY").befund == ["tot"]
    assert _eintrag(inv, "PS_SET").befund == ["tot"]
    assert _eintrag(inv, "PS_SET").gesetzt == ["aussen/skripte/start.ps1:2"]
    assert _eintrag(inv, "PS_KLAMMER").gesetzt == ["aussen/skripte/start.ps1:4"]
    assert _eintrag(inv, "ONLY_READ").befund == ["ungesetzt"]
    assert _eintrag(inv, "PS_READ").befund == ["ungesetzt"]
    assert _eintrag(inv, "PS_CMP").gelesen == ["aussen/skripte/start.ps1:5"]
    assert _eintrag(inv, "FROM_HOST").befund == ["ungesetzt"]
    assert _eintrag(inv, "FOO").befund == ["doppelt"]  # .env.example und Compose
    assert _eintrag(inv, "LISTED").befund == []
    assert _eintrag(inv, "BAR_KEY").befund == []
    assert _eintrag(inv, "PATH").befund == []  # Systemname: nie "ungesetzt"
    namen = {n.name for n in inv.namen}
    assert "KOMMENTIERT" not in namen
    # aus dem Graphen nur echte Lesestellen, nicht die vermischten Dateiarten
    assert not namen & {"NUR_IM_BEISPIEL_GRAPH", "NUR_IN_COMPOSE_GRAPH", "NUR_IN_PS_GRAPH", "NUR_IN_ENVLOCAL_GRAPH"}
    text = inventar_fund(inv).text
    assert "Konsolidierungsplan" in text
    assert "1 doppelt gesetzte Namen in eine Quelle" in text


def test_graph_fehlt_notiz_kein_absturz(git_repo, tmp_path: Path) -> None:
    wurzel = _repo(git_repo)
    leer = tmp_path / "leerer-stand"
    leer.mkdir()
    inv = inventar(wurzel, stand_unter=leer)
    assert any(n.startswith("quelle-unlesbar:graph") for n in inv.notizen)
    assert _eintrag(inv, "FOO").gesetzt  # Dateiquellen trotzdem da
    assert all("tot" not in n.befund for n in inv.namen)  # ohne Lesestellen kein "tot"
    (leer / "graph.sqlite").write_bytes(b"kein sqlite")
    inv2 = inventar(wurzel, stand_unter=leer)
    assert any(n.startswith("quelle-unlesbar:graph") for n in inv2.notizen)


def test_deterministisch(git_repo, graph_stand) -> None:
    wurzel = _repo(git_repo)
    stand = _graph(graph_stand)
    a, b = inventar(wurzel, stand_unter=stand), inventar(wurzel, stand_unter=stand)
    assert inventar_fund(a).text.encode() == inventar_fund(b).text.encode()
    assert inventar_tabelle(a) == inventar_tabelle(b)
    assert a == b
    for n in a.namen:
        assert n.gelesen == sorted(n.gelesen) and n.gesetzt == sorted(n.gesetzt) and n.befund == sorted(n.befund)


def test_anker_stabil(git_repo, graph_stand) -> None:
    wurzel = _repo(git_repo)
    f1 = inventar_fund(inventar(wurzel, stand_unter=_graph(graph_stand)))
    git_repo.datei(".env.example", "NEU=1\n")
    git_repo.commit("mehr")
    f2 = inventar_fund(inventar(wurzel, stand_unter=_graph(graph_stand, _kante("X", "aussen", "a.py", 99))))
    assert f1.art == f2.art == "konfig"
    assert f1.schwere == f2.schwere == "mittel"
    assert f1.code_repo == f2.code_repo == "aussen"
    assert f1.anker == f2.anker == "konfig-inventar:aussen"
    assert f1.text != f2.text


def test_schluessel_artiger_name(git_repo, graph_stand) -> None:
    git_repo.datei("selbstbild.yaml", "name: aussen\n")
    token = "ghp_" + "A1b2C3d4E5" * 4
    git_repo.datei(".env.example", f"{token}=x\nsk-abcdefghijklmnopqrstuvwxyz0123=y\nNORMAL=z\n")
    git_repo.commit("init")
    inv = inventar(git_repo.wurzel, stand_unter=graph_stand([], []))
    f = inventar_fund(inv)
    ausgabe = _alle_ausgaben(inv)
    assert token not in ausgabe and "sk-abc" not in ausgabe
    assert pruefen(f.text) is None and pruefen(inventar_tabelle(inv)) is None
    assert {n.name for n in inv.namen} == {"NORMAL"}
    # nur das Muster wird genannt, nie der Name
    assert any(n.startswith("schluessel-verdacht:") and "GitHub-Token" in n for n in inv.notizen)


def test_kappe_und_volle_tabelle(git_repo, graph_stand) -> None:
    git_repo.datei("selbstbild.yaml", "name: aussen\n")
    git_repo.datei(".env.example", "".join(f"NAME_{i:05d}_MIT_LANGEM_SUFFIX=1\n" for i in range(4000)))
    git_repo.commit("viele")
    inv = inventar(git_repo.wurzel, stand_unter=graph_stand([], []))
    text = inventar_fund(inv).text
    assert len(text) < 65536
    assert "weitere, siehe lokaler Bericht" in text
    k = int(text.split("… und ")[1].split(" weitere")[0])
    assert kappen_zeile(k) in text
    tabelle = inventar_tabelle(inv)
    assert "NAME_03999_MIT_LANGEM_SUFFIX" in tabelle and "weitere, siehe" not in tabelle


def test_submodul_eigener_repo_name(git_repo, graph_stand, tmp_path: Path) -> None:
    import os
    sub = type(git_repo)(tmp_path / "subquelle", dict(os.environ))
    sub.datei(".env.sample", "SUB_NAME=subwert\n")
    sub.datei("secrets.env", "SUB_NAME=subgeheim\n")
    sub.commit("s")
    git_repo.datei("selbstbild.yaml", "name: aussen\n")
    git_repo.datei(".env.example", "SUB_NAME=1\n")
    git_repo.submodul("sub", sub)
    git_repo.commit("mit sub")
    inv = inventar(git_repo.wurzel, stand_unter=graph_stand([], []))
    eintrag = _eintrag(inv, "SUB_NAME")
    assert eintrag.gesetzt == ["aussen/.env.example:1", "aussen/sub/.env.sample:1"]
    assert eintrag.befund == ["doppelt", "tot"]
    ausgabe = _alle_ausgaben(inv)
    assert "subwert" not in ausgabe and "subgeheim" not in ausgabe
