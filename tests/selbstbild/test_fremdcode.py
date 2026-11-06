from selbstbild.fremdcode import fremd_pruefer, ist_fremd_id

P = fremd_pruefer(["vibemind-os/openclaw/**", "**/external/**"], "vibemind")


def test_submodul_repo_trifft():
    assert P("vibemind/vibemind-os/openclaw", "extensions/matrix/src/a.ts")


def test_aeusseres_repo_pfad_ohne_praefix():
    assert P("vibemind", "lib/external/x.py") and not P("vibemind", "lib/intern/x.py")


def test_segmentgrenze():
    assert not P("vibemind/vibemind-os/openclaw-tools", "x.py")


def test_fremdes_aussen_nie():
    assert not P("anderes/vibemind-os/openclaw", "x.py")


def test_leer_nie_fremd():
    assert not fremd_pruefer([], "vibemind")("vibemind/vibemind-os/openclaw", "x.py")


def test_ist_fremd_id():
    assert ist_fremd_id(P, "symbol:vibemind/vibemind-os/openclaw:a/b.ts#f")
    assert ist_fremd_id(P, "datei:vibemind/vibemind-os/openclaw:a/b.ts")
    assert not ist_fremd_id(P, "symbol:vibemind/vibemind-os/coding-engine:a.py#f")
    assert not ist_fremd_id(P, "doku:vibemind:x.md")


def test_ist_fremd_id_raute_im_pfad():
    assert ist_fremd_id(P, "symbol:vibemind:lib/a#b/external/x.py#f")  # letzte Raute trennt den Qualifizierer
    assert ist_fremd_id(P, "datei:vibemind:lib/a#b/external/x.py")  # Datei-Id: kein Qualifizierer
    assert not ist_fremd_id(P, "symbol:vibemind:lib/a#b/intern/x.py#f")
