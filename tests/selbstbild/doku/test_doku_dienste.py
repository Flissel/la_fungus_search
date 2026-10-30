"""doku_loop: Dienst-Register."""

import json
import sys
from pathlib import Path

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import dienste  # noqa: E402
from selbstbild.doku.dienste import Dienst, DienstRegister  # noqa: E402
from selbstbild.doku.extraktoren.compose import ComposeService  # noqa: E402


def test_register_aufloesen_und_unbekannte(tmp_path):
    pfad = tmp_path / "d.yaml"
    pfad.write_text(
        "dienste:\n  zeta:\n    port: 1\n    ort: vm\n    space: null\n    env: [Z_URL]\n"
        "  alpha:\n    port: null\n    ort: extern\n    space: marketing\n    env: [A_URL, B_URL]\n",
        encoding="utf-8")
    reg = DienstRegister.laden(pfad)
    assert reg.aufloesen("B_URL").name == "alpha"
    assert reg.aufloesen("b_url") is None
    assert reg.unbekannte(["A_URL", "NIX", "Z_URL", "NIX"]) == ["NIX"]
    aus = tmp_path / "o.yaml"
    reg.speichern(aus)
    erster = aus.read_bytes()
    DienstRegister.laden(aus).speichern(aus)
    assert aus.read_bytes() == erster
    assert erster.index(b"alpha") < erster.index(b"zeta")


def test_env_in_zwei_diensten_ist_fehler(tmp_path):
    pfad = tmp_path / "d.yaml"
    pfad.write_text(
        "dienste:\n  a: {port: 1, ort: vm, space: null, env: [X_URL]}\n"
        "  b: {port: 2, ort: vm, space: null, env: [X_URL]}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        DienstRegister.laden(pfad)


def test_unbekannter_ort_ist_fehler():
    with pytest.raises(ValueError):
        DienstRegister({"a": Dienst("a", 1, "mond", None, ())})


def test_seed_aus_compose_enthaelt_openfang_aus_ports_json(tmp_path):
    ports = tmp_path / "vibemind.config.json"
    ports.write_text(json.dumps({"ports": {"openfang": 4200, "qdrant": 1}}), encoding="utf-8")
    services = [
        ComposeService("s.yml", "qdrant", (6333,), {}),
        ComposeService("s.yml", "brain-core", (), {"QDRANT_URL": "http://qdrant:6333", "X": "nix"}),
    ]
    reg = dienste.seed_aus_compose(services, ports)
    assert reg.dienste["openfang"].ort == "pc" and reg.dienste["openfang"].port == 4200
    assert reg.dienste["qdrant"].ort == "container" and reg.dienste["qdrant"].port == 6333
    assert reg.aufloesen("QDRANT_URL").name == "qdrant"
    assert reg.aufloesen("X") is None


def test_seed_nackter_name_nur_bei_host_schluessel(tmp_path):
    ports = tmp_path / "p.json"
    ports.write_text(json.dumps({"ports": {}}), encoding="utf-8")
    services = [
        ComposeService("s.yml", "postgres", (5432,), {"POSTGRES_USER": "postgres"}),
        ComposeService("s.yml", "app", (), {"POSTGRES_PASSWORD": "postgres", "DB_HOST": "postgres"}),
    ]
    reg = dienste.seed_aus_compose(services, ports)
    assert reg.dienste["postgres"].env == ("DB_HOST",)
    assert reg.aufloesen("POSTGRES_PASSWORD") is None and reg.aufloesen("POSTGRES_USER") is None


@braucht_vm
def test_committetes_register_ladbar_und_aufloesbar():
    reg = DienstRegister.laden(REPO / "config" / "dienste.yaml")
    erwartet = {"MARKETING_PULT_URL": "marketing-api", "MARKETING_HTTP_URL": "marketing-api",
                "OLLAMA_URL": "ollama", "MIROFISH_URL": "mirofish", "LAURA_API_URL": "laura-api",
                "TELEGRAM_API_BASIS": "telegram"}
    for env, name in erwartet.items():
        assert reg.aufloesen(env).name == name


def test_seed_setzt_space_aus_compose_pfad_und_brain_praefix(tmp_path):
    ports = tmp_path / "p.json"
    ports.write_text(json.dumps({"ports": {"brain_dashboard": 1}}), encoding="utf-8")
    services = [
        ComposeService("vibemind-os/spaces/sales-claw/docker-compose.yml", "sales-mail", (1,), {}),
        ComposeService("vibemind-os/spaces/learning/deployment/compose.yml", "learning-api", (2,), {}),
        ComposeService("infra/swarm/vibemind-stack.yml", "brain-core", (5000,), {}),
        ComposeService("infra/swarm/vibemind-stack.yml", "qdrant", (6333,), {}),
        ComposeService("vibemind-os/spaces/learning/deployment/compose.yml", "qdrant", (6334,), {}),
        ComposeService("vibemind-os/spaces/learning/deployment/compose.yml", "brain-x", (), {}),
        ComposeService("vibemind-os/spaces/learning/deployment/compose.yml", "redis", (), {}),
        ComposeService("vibemind-os/spaces/learning/deployment/compose.yml", "redis", (7,), {}),
    ]
    reg = dienste.seed_aus_compose(services, ports)
    assert reg.dienste["sales-mail"].space == "sales-claw"
    assert reg.dienste["learning-api"].space == "learning"
    assert reg.dienste["brain-core"].space == "brain"
    assert reg.dienste["qdrant"].space is None  # zwei Dateien, verschiedene Spaces -> keiner
    assert reg.dienste["redis"].space == "learning"  # zweimal dieselbe Datei
    assert reg.dienste["brain-x"].space == "learning"  # Space-Pfad vor brain-Praefix
    assert reg.dienste["brain_dashboard"].space is None  # nur Compose-Dienste


@braucht_vm
def test_committetes_register_marketing_api_im_space_marketing():
    reg = DienstRegister.laden(REPO / "config" / "dienste.yaml")
    assert reg.dienste["marketing-api"].space == "marketing"
