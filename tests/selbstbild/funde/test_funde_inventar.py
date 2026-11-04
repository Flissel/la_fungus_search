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
         "secretsgeheim", "PEMGEHEIM", "hostgeheim", "grossgeheim", "secrets2geheim", "envrcgeheim")
VERBOTEN = (".env", ".env.local", "secrets.env", "cert.pem", ".ENV", "Secrets.env", ".envrc")

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
        # S5-13: nur nicht-produktive Lesestellen -> nie "ungesetzt"
        _kante("NUR_IN_TESTS", "aussen", "tests/test_a.py", 1),
        _kante("NUR_VENDOR", "aussen", "lib/vendor/x.py", 1),
        _kante("NUR_SITE", "aussen", "venv2/site-packages/y.py", 1),
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
    # Windows: Gross-/Kleinschreibung kollidiert im selben Ordner, darum eigene Ordner
    r.datei("gross/.ENV", "FOO=grossgeheim\n")
    r.datei("andere/Secrets.env", "FOO=secrets2geheim\n")
    r.datei(".envrc", "export FOO=envrcgeheim\n")
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
    for name in (".env.production", ".env.sanitized", "prod.env", "example.env", "key.PEM", ".ENV", "Secrets.env",
                 ".envrc", "env.py", "server.key", "id_rsa.pub", "credentials.json", "secrets.yaml"):
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
    for name in ("NUR_IN_TESTS", "NUR_VENDOR", "NUR_SITE"):  # S5-13: nur nicht-produktive Lesestellen
        assert _eintrag(inv, name).gelesen and _eintrag(inv, name).befund == []
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
    assert f1.titel == f2.titel == "Konfig-Inventar"  # S5-11: das Einzel-Issue bearbeitet nur den Text
    assert f1.text != f2.text


def test_schluessel_artiger_name(git_repo, graph_stand) -> None:
    git_repo.datei("selbstbild.yaml", "name: aussen\n")
    token = "ghp_" + "A1b2C3d4E5" * 4
    git_repo.datei(".env.example", f"{token}=x\nsk-abcdefghijklmnopqrstuvwxyz0123=y\nNORMAL=z\n")
    git_repo.commit("init")
    inv = inventar(git_repo.wurzel, stand_unter=graph_stand([], []))
    f = inventar_fund(inv)
    ausgabe = _alle_ausgaben(inv)
    assert token not in ausgabe and token.upper() not in ausgabe and "sk-abc" not in ausgabe
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


# ---------------------------------------------------------------- Fix-Runde 1

def _klein(git_repo, dateien: dict[str, str], yaml_text: str = "name: aussen\n") -> Path:
    git_repo.datei("selbstbild.yaml", yaml_text)
    for pfad, inhalt in dateien.items():
        git_repo.datei(pfad, inhalt)
    git_repo.commit("klein")
    return git_repo.wurzel


def test_mehrzeilige_werte_keine_namen(git_repo, graph_stand) -> None:
    """Important (D6): Fragmente mehrzeiliger Werte werden nie Namen."""
    beispiel = (
        'CERT="-----BEGIN CERTIFICATE-----\n'
        "MIIBszCCAVmgAwIBAgIUxyz012=\n"
        '-----END CERTIFICATE-----"\n'
        "JSON='{\n"
        "secretpart=abc\n"
        "}'\n"
        'ESC="a\\"b\n'
        "INNEN_ESC=1\n"
        '"\n'
        "DANACH=1\n")
    wurzel = _klein(git_repo, {".env.example": beispiel, "b.example.env": 'OK=1\nOFFEN="abc\nNACH_OFFEN=1\n'})
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    assert {n.name for n in inv.namen} == {"CERT", "JSON", "ESC", "DANACH", "OK", "OFFEN"}
    ausgabe = _alle_ausgaben(inv)
    for fragment in ("MIIBszCC", "MIIBSZCC", "xyz012", "XYZ012", "secretpart", "SECRETPART", "INNEN_ESC",
                     "NACH_OFFEN", "abc"):
        assert fragment not in ausgabe, fragment
    assert any("offenem Anführungszeichen" in n and n.startswith("1 ") for n in inv.notizen)


def test_compose_merge_explizit_gewinnt(git_repo, graph_stand) -> None:
    compose = (
        "x-basis: &basis\n"
        "  image: i\n"
        "  environment:\n"
        "    NUR_BASIS: a\n"
        "services:\n"
        "  eigen:\n"
        "    <<: *basis\n"
        "    environment:\n"
        "      EIGEN: b\n"
        "  geerbt:\n"
        "    <<: *basis\n"
        "  env_merge:\n"
        "    environment:\n"
        "      <<: {AUS_MERGE: c}\n"
        "      EXPLIZIT: d\n")
    wurzel = _klein(git_repo, {"docker-compose.yml": compose})
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    namen = {n.name: n for n in inv.namen if n.art in ("env", "key")}
    assert set(namen) == {"EIGEN", "NUR_BASIS", "AUS_MERGE", "EXPLIZIT"}
    # NUR_BASIS kommt nur aus `geerbt` (eine Stelle: die Anker-Definition), `eigen` ueberschreibt environment
    assert namen["NUR_BASIS"].gesetzt == ["aussen/docker-compose.yml:4"]
    # nur `eigen` ohne `geerbt`: NUR_BASIS gibt es dann gar nicht
    git_repo.datei("docker-compose.yml", compose.split("  geerbt:")[0])
    git_repo.commit("ohne geerbt")
    inv2 = inventar(wurzel, stand_unter=graph_stand([], []))
    assert "NUR_BASIS" not in {n.name for n in inv2.namen} and "EIGEN" in {n.name for n in inv2.namen}


def test_compose_dollar_fragmente(git_repo, graph_stand) -> None:
    compose = (
        "services:\n"
        "  app:\n"
        "    command: \"echo $NICHT_GANZ and ${KLAMMER?fehlt}\"\n"
        "    image: \"r/${TAG:-latest}\"\n"
        "    environment:\n"
        "      - HASH=$2y$10$N9qo8uLOickgx2ZMRZoMye\n"
        "      - LISTE=$LISTE_GANZ\n"
        "  b:\n"
        "    environment:\n"
        "      GANZ: $GANZ_REF\n"
        "      TEIL: pre$TEIL_REF\n"
        "      ESC: $${NICHT_ESC}\n")
    wurzel = _klein(git_repo, {"docker-compose.yml": compose})
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    gelesen = {n.name for n in inv.namen if n.gelesen}
    assert gelesen == {"KLAMMER", "TAG", "LISTE_GANZ", "GANZ_REF"}
    ausgabe = _alle_ausgaben(inv).upper()
    for fragment in ("N9QO8U", "NICHT_GANZ", "TEIL_REF", "NICHT_ESC", "LATEST", "FEHLT`"):
        assert fragment not in ausgabe, fragment


def test_powershell_setzformen(git_repo, graph_stand) -> None:
    ps = (
        '$env:NULLISH ??= "x"\n'
        '[Environment]::SetEnvironmentVariable("SETENV_A", "v", "User")\n'
        "[System.Environment]::SetEnvironmentVariable('SETENV_B', $env:LIES_B)\n"
        'Set-Item env:ITEM_A "v"\n'
        'Set-Item -Path env:ITEM_B -Value "v"\n'
        'New-Item env:ITEM_C -Value "v"\n'
        "<# Block\n"
        "$env:IM_BLOCK = 1\n"
        "#>\n"
        'Write-Host "x" # $env:IM_ZEILENKOMMENTAR\n'
        'Write-Host "#$env:IM_STRING"\n'
        "<# inline #> $env:NACH_BLOCK = 1\n")
    wurzel = _klein(git_repo, {"s.ps1": ps, "m.psm1": "$env:IM_MODUL = 1\n"})
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    gesetzt = {n.name for n in inv.namen if n.gesetzt}
    gelesen = {n.name for n in inv.namen if n.gelesen}
    assert gesetzt == {"NULLISH", "SETENV_A", "SETENV_B", "ITEM_A", "ITEM_B", "ITEM_C", "NACH_BLOCK", "IM_MODUL"}
    assert gelesen == {"LIES_B", "IM_STRING"}
    assert _eintrag(inv, "NACH_BLOCK").gesetzt == ["aussen/s.ps1:12"]


def test_gross_klein_ein_eintrag(git_repo, graph_stand) -> None:
    wurzel = _klein(git_repo, {"s.ps1": "$env:Foo = 1\n"})
    stand = graph_stand([], [_kante("FOO", "aussen", "app.py", 2), _kante("foo", "aussen", "b.py", 3)])
    inv = inventar(wurzel, stand_unter=stand)
    assert [n.name for n in inv.namen if n.art == "env"] == ["FOO"]
    foo = _eintrag(inv, "FOO")
    assert foo.gesetzt == ["aussen/s.ps1:1"] and foo.gelesen == ["aussen/app.py:2", "aussen/b.py:3"]
    assert foo.befund == []


def test_submodul_nicht_ausgecheckt(git_repo, graph_stand) -> None:
    from selbstbild.funde.sync import unlesbare_arten
    wurzel = _klein(git_repo, {".gitmodules": '[submodule "fehlt"]\n\tpath = fehlt\n\turl = ../fehlt\n',
                               ".env.example": "A=1\n"})
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    assert "quelle-unlesbar:submodul: aussen/fehlt nicht ausgecheckt" in inv.notizen
    assert inv.fehlende_submodule == ["aussen/fehlt"]
    text = inventar_fund(inv).text
    assert text.index("Unvollständig") < text.index("Konsolidierungsplan") and "`aussen/fehlt`" in text
    assert unlesbare_arten(inv.notizen) == frozenset({"konfig"})


def test_dienste_unlesbar_faelle(git_repo, graph_stand) -> None:
    yaml_mit = "name: aussen\ndienste: config/dienste.yaml\n"
    wurzel = _klein(git_repo, {"config/dienste.yaml": "dienste:\n  a: [unvollstaendig\n"}, yaml_mit)
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    assert any(n.startswith("quelle-unlesbar:dienste:") for n in inv.notizen)
    assert not [n for n in inv.namen if n.art == "port"]
    # zu gross
    gross = "dienste:\n" + "".join(f"  d{i}:\n    port: {1000 + i}\n" for i in range(200))
    _klein(git_repo, {"config/dienste.yaml": gross}, yaml_mit + "max_dateigroesse_kb: 1\n")
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    assert any(n.startswith("quelle-unlesbar:dienste:") for n in inv.notizen)
    assert not [n for n in inv.namen if n.art == "port"]
    # konfiguriert, aber nicht versioniert
    git_repo.git("rm", "-q", "config/dienste.yaml")
    _klein(git_repo, {}, yaml_mit)
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    assert any(n.startswith("quelle-unlesbar:dienste:") for n in inv.notizen)
    # nicht konfiguriert und nicht da: nur ein Hinweis, keine Unlesbarkeit
    _klein(git_repo, {}, "name: aussen\n")
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    assert not any(n.startswith("quelle-unlesbar:") for n in inv.notizen)


def test_doppelter_dienst_schluessel(git_repo, graph_stand) -> None:
    dienste = "dienste:\n  alpha:\n    port: 1111\n  alpha:\n    port: 2222\n"
    wurzel = _klein(git_repo, {"config/dienste.yaml": dienste})
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    for port in ("1111", "2222"):
        assert _eintrag(inv, port).befund == ["widerspruechlich: Dienst alpha mit Ports 1111, 2222"]


def test_compose_kaputt_kein_absturz(git_repo, graph_stand) -> None:
    wurzel = _klein(git_repo, {"docker-compose.yml": "services:\n  a: [kaputt\n", ".env.example": "A=1\n"})
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    assert inv.uebersprungen == 1 and any("1 Compose-Datei(en)" in n for n in inv.notizen)
    assert _eintrag(inv, "A").gesetzt == ["aussen/.env.example:1"]


@pytest.mark.parametrize("max_zeichen", [120, 600, 2000])
def test_kappe_klein_haelt_grenze(git_repo, graph_stand, max_zeichen: int) -> None:
    wurzel = _klein(git_repo, {".env.example": "".join(f"N_{i}=1\n" for i in range(300))})
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    text = inventar_fund(inv, max_zeichen=max_zeichen).text
    assert len(text) <= max_zeichen
    assert "weitere, siehe lokaler Bericht" in text


@pytest.mark.parametrize("fehler", ["fehlt", "zeit", "exit"])
def test_git_scheitert_kein_absturz(git_repo, graph_stand, monkeypatch: pytest.MonkeyPatch, fehler: str) -> None:
    import subprocess

    from selbstbild.funde import inventar as modul
    wurzel = _klein(git_repo, {".env.example": "A=1\n"})

    def lauf(args, **kw):
        assert kw.get("timeout") == 60
        if fehler == "fehlt":
            raise FileNotFoundError("git")
        if fehler == "zeit":
            raise subprocess.TimeoutExpired(args, 60)
        return subprocess.CompletedProcess(args, 128, b"", b"fatal")

    monkeypatch.setattr(modul.subprocess, "run", lauf)
    inv = inventar(wurzel, stand_unter=graph_stand([], []))
    assert any(n.startswith("quelle-unlesbar:dateien: git ls-files fehlgeschlagen für aussen") for n in inv.notizen)
    assert not any(n.name == "A" for n in inv.namen)
