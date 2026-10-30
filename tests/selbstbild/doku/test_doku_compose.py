"""doku_loop: Compose-Extraktor."""

import dataclasses
import sys
from pathlib import Path

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku.extraktoren import compose  # noqa: E402


STACK = """
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
  sales-stt:
    image: x
    ports:
      - "127.0.0.1:2790:2790"
      - {published: 2791, target: 80}
      - 2792
      - "kaputt"
  brain-core:
    environment:
      QDRANT_URL: http://qdrant:6333
      OHNE_VORGABE: ${NIX}
      QDRANT_HOST: qdrant
      EXTERN: https://example.org/x
  sales-mcp:
    environment:
      - STT_URL=${STT_URL:-http://sales-stt:2790}
      - ANDERS=${ANDERS-http://qdrant:6333/}
"""


def _konfig(tmp_path, dateien):
    k = dataclasses.replace(vm_laden(), repo=tmp_path)
    return dataclasses.replace(k, compose_live=tuple(dateien))


def test_services_ports_env_und_kanten(tmp_path):
    (tmp_path / "s.yml").write_text(STACK, encoding="utf-8")
    services, kanten = compose.compose_lesen(_konfig(tmp_path, ["s.yml"]))
    by = {s.name: s for s in services}
    assert by["qdrant"].ports == (6333,)
    assert by["sales-stt"].ports == (2790, 2791, 2792)
    assert by["brain-core"].env["OHNE_VORGABE"] == ""
    assert by["sales-mcp"].env["STT_URL"] == "http://sales-stt:2790"
    paare = {(k.von_service, k.env, k.nach_service) for k in kanten}
    assert paare == {
        ("brain-core", "QDRANT_URL", "qdrant"),
        ("brain-core", "QDRANT_HOST", "qdrant"),
        ("sales-mcp", "STT_URL", "sales-stt"),
        ("sales-mcp", "ANDERS", "qdrant"),
    }
    assert all(k.datei == "s.yml" for k in kanten)


def test_rendered_dateien_werden_ignoriert(tmp_path):
    (tmp_path / ".vibemind-stack-rendered-1.yml").write_text(STACK, encoding="utf-8")
    with pytest.raises(ValueError):
        compose.compose_lesen(_konfig(tmp_path, [".vibemind-stack-rendered-1.yml"]))
    with pytest.raises(ValueError):
        compose.compose_lesen(_konfig(tmp_path, ["x/partial-a.yml"]))


def test_fehlende_datei_wird_uebersprungen(tmp_path):
    assert compose.compose_lesen(_konfig(tmp_path, ["weg.yml"])) == ([], [])


@braucht_vm
def test_echtes_repo():
    services, kanten = compose.compose_lesen(vm_laden())
    assert any(s.datei == "infra/swarm/vibemind-stack.yml" for s in services)
    assert all("\\" not in s.datei for s in services)


def test_nackter_name_nur_bei_host_artigem_schluessel_und_keine_selbstkante(tmp_path):
    (tmp_path / "s.yml").write_text(
        "services:\n  postgres:\n    environment:\n      SELF_HOST: postgres\n"
        "  app:\n    environment:\n      POSTGRES_PASSWORD: postgres\n      POSTGRES_USER: postgres\n"
        "      DB_HOST: postgres\n      SELF_URL: http://app:1/\n", encoding="utf-8")
    _, kanten = compose.compose_lesen(_konfig(tmp_path, ["s.yml"]))
    assert [(k.von_service, k.env, k.nach_service) for k in kanten] == [("app", "DB_HOST", "postgres")]
