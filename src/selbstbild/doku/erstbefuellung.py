"""Erstbefuellung der Doku (Stufe 4, Task 7, D10, Spec 12a).

Heute wird eine Aussage nur beurteilt, wenn sich ihr Beleg aendert. Die Erstbefuellung prueft jede gebundene Aussage
EINMAL gegen den heutigen (committeten) Code: Judge-Fragetyp `erstpruefung` ("Stimmt die Aussage zum Code?", alter
gleich neuer Quelltext) mit `judge.zweimal_einig`, unter dem eigenen Budget-Agenten (Vorgabe
`selbstbild-erstbefuellung`, 150/Tag in `config/agent_budget.yaml`), nur im Fenster `[von, bis)` Ortszeit (Vorgabe
00:00 bis 06:00) und nur, wenn der Budget-Waechter zustimmt (Klasse `hintergrund`: Pause bei User-Arbeit). Sie endet
von selbst, sobald nichts mehr offen ist (`Stand.erstbefuellung_fertig`); danach gilt wieder nur `doku-loop`.

Auswahl (Ruling T7-a): nur Aussagen mit mindestens einem signaltragenden Beleg (`tick.signal_traegt`); eine Aussage,
die nur ganze-Datei-Anker auf Code-Dateien binden, wird nicht geprueft (sie steht gesammelt in der Pruefliste).
Reihenfolge deterministisch nach Datei, dann Zeile. Befragt wird der erste signaltragende Beleg der Aussage.

Abbruch: bei einer Budget-Ablehnung, jedem WIEDERHOLEN-Grund (`tick.WIEDERHOLEN`: Budget, Timeout, Exit-Code, claude
fehlt, OSError) und an der Tick-Deadline. Scheitert in `zweimal_einig` erst der zweite Aufruf, gilt die Aussage als
NICHT geprueft und der naechste Lauf fragt beide neu: das kostet genau eine Buchung extra (die verworfene erste),
hoechstens einmal je Lauf, weil der Schritt danach abbricht. Inhaltliche Fehlschlaege (`uneinig`, `ungueltig`,
`prompt:*`) zaehlen als geprueft und werden Pruefpunkt.

Urteils-Cache (`stand.urteil_holen/urteil_merken`, Fragetyp `erstpruefung`): Schluessel `["erstpruefung", aussage_id,
art, datei, anker, digest]` mit `neu_digest = digest` (alt = neu = heutiger Digest). Ein spaeteres echtes Signal auf
demselben Anker sucht unter dem Signal-Schluessel (ohne Fragetyp) mit einem anderen `neu_digest` und fragt deshalb
immer neu; ein Erst-Urteil beantwortet nie ein Signal und umgekehrt.

Dieser Modul fragt nur und meldet die Urteile; das Schreiben (`veraltet` mit gueltigem Ersatz ueber denselben Weg wie
ein Signal: CAS, Torwaechter, Ledger, pathspec-Commit), die Pruefpunkte und den Stand macht der Tick
(`tick._Lauf._erstbefuellung`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, time
from typing import TYPE_CHECKING, Any

from .anker import Beleg
from .aussagen import Aussage
from .konfig import Konfig

if TYPE_CHECKING:
    from .graph import Graph
    from .judge import JudgeAntwort
    from .stand import Stand

__all__ = [
    "AGENT_VORGABE", "FRAGETYP", "ErstEinstellung", "ErstErgebnis", "ErstPosten", "ErstUrteil", "einstellung",
    "erstbefuellung_faellig", "erstbefuellung_schritt", "im_fenster", "posten_waehlen", "stand_fortschreiben",
]

AGENT_VORGABE = "selbstbild-erstbefuellung"
VON_VORGABE = "00:00"
BIS_VORGABE = "06:00"
FRAGETYP = "erstpruefung"


@dataclass(frozen=True)
class ErstEinstellung:
    aktiv: bool
    von: time
    bis: time
    agent: str


def _uhrzeit(text: str) -> time:
    stunde, minute = text.split(":")
    return time(int(stunde), int(minute))


def einstellung(konfig: Konfig) -> ErstEinstellung:
    """`konfig.erstbefuellung` (in `konfig._erstbefuellung_pruefen` validiert) mit den Vorgaben."""
    roh = konfig.erstbefuellung or {}
    return ErstEinstellung(roh.get("aktiv") is True, _uhrzeit(str(roh.get("von", VON_VORGABE))),
                           _uhrzeit(str(roh.get("bis", BIS_VORGABE))), str(roh.get("agent") or AGENT_VORGABE))


def im_fenster(e: ErstEinstellung, jetzt: datetime) -> bool:
    """Ortszeit von `jetzt` in `[von, bis)`; ein Fenster ueber Mitternacht (`von > bis`) gilt bis `bis` am Folgetag.
    Zeitzonenbewusste Zeiten zaehlen in der Ortszeit des Rechners, naive gelten schon als Ortszeit."""
    ort = jetzt.astimezone() if jetzt.tzinfo is not None else jetzt
    t = ort.time()
    if e.von <= e.bis:
        return e.von <= t < e.bis
    return t >= e.von or t < e.bis


def erstbefuellung_faellig(konfig: Konfig, stand: Stand | None, jetzt: datetime) -> bool:
    """Wahr, wenn die Erstbefuellung aktiv ist, noch nicht fertig und die Ortszeit im Fenster liegt."""
    e = einstellung(konfig)
    if not e.aktiv or (stand is not None and stand.erstbefuellung_fertig):
        return False
    return im_fenster(e, jetzt)


@dataclass(frozen=True)
class ErstPosten:
    """Eine Aussage fuer die Erstpruefung. `aussage.belege` sind ihre gemerkten Belege (daraus die erlaubten Anker),
    `beleg` der befragte (erste signaltragende). `fragbar=False`: in diesem Lauf nicht fragen (echtes Signal im selben
    Lauf, Datei schon umgeschrieben, Beleg mit uncommitteten Aenderungen) -- zaehlt aber als offen."""
    aussage: Aussage
    beleg: Beleg
    fragbar: bool = True
    nur_lesend: bool = False


@dataclass(frozen=True)
class ErstUrteil:
    posten: ErstPosten
    antwort: JudgeAntwort | None
    grund: str  # ok | uneinig | ungueltig | prompt:<Typ>
    aus_cache: bool = False


@dataclass(frozen=True)
class ErstErgebnis:
    geprueft: int  # Aussagen mit Urteil in diesem Schritt (frisch oder aus dem Cache)
    offen: int  # danach noch nicht geprueft
    fertig: bool
    urteile: tuple[ErstUrteil, ...] = ()
    abbruch: str | None = None  # budget:<grund> | timeout | exit:<n> | nicht_gefunden | oserror:<Typ> | deadline


def posten_waehlen(konfig: Konfig, eintraege: Iterable[tuple[Aussage, Sequence[Beleg], bool, bool]]
                   ) -> list[ErstPosten]:
    """Aus `(aussage, gemerkte Belege, fragbar, nur_lesend)` die Posten mit mindestens einem signaltragenden Beleg
    (Ruling T7-a), nach Datei, Zeile und Id sortiert; doppelte Ids zaehlen einmal (der erste Eintrag)."""
    from .tick import signal_traegt  # tick importiert dieses Modul

    posten: dict[str, ErstPosten] = {}
    for a, belege, fragbar, nur_lesend in eintraege:
        if a.id in posten:
            continue
        tragend = [b for b in belege if signal_traegt(b, a.datei, konfig)]
        if not tragend:
            continue
        posten[a.id] = ErstPosten(dataclasses.replace(a, belege=tuple(belege)), tragend[0], fragbar, nur_lesend)
    return sorted(posten.values(), key=lambda p: (p.aussage.datei, p.aussage.zeile_start, p.aussage.id))


def _quelltext_datei(konfig: Konfig) -> Callable[[Beleg], str]:
    def lesen(b: Beleg) -> str:
        try:
            return (konfig.repo / b.file).read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return "(Datei fehlt)"
    return lesen


def erstbefuellung_schritt(konfig: Konfig, graph: Graph | None, stand: Stand | None, *, budget: Any,
                           laufzeit: Callable[..., Any], deadline: Callable[[], bool], posten: Sequence[ErstPosten],
                           quelltext: Callable[[Beleg], str] | None = None,
                           digest: Callable[[Beleg], str | None] | None = None,
                           nachbarn: Callable[[Beleg], tuple[str, ...]] | None = None,
                           offene_seiten: int = 0, vollstaendig: bool = True,
                           jetzt: datetime | None = None) -> ErstErgebnis:
    """Ein Schritt der Erstbefuellung ueber `posten` (siehe Moduldoku). Gefragt werden die noch nicht erstgeprueften
    (`stand.erstgeprueft`), fragbaren Posten in Datei/Zeile-Reihenfolge, je Posten `zweimal_einig` unter dem Agenten
    der Erstbefuellung; Cache zuerst. `deadline()` wahr -> Abbruch vor dem naechsten Posten. Der Stand wird nicht
    veraendert (`stand_fortschreiben`).

    `quelltext(beleg)`: heutiger Quelltext des Belegs (Tick: aus dem Messbaum; Vorgabe: die Datei unter
    `konfig.repo`); `digest(beleg)`: heutiger Digest (Vorgabe: `beleg.digest`); `nachbarn(beleg)`: Graph-Nachbarn fuer
    den Prompt (Vorgabe: keine; `graph` wird nur dafuer gebraucht).

    `fertig` = `vollstaendig` (alle Doku-Dateien gelesen) und nichts offen und `offene_seiten == 0`. Plan-Ruling P3:
    Task 9 uebergibt hier die Zahl der noch nie geschriebenen Rowboat-Seiten."""
    from . import judge  # nie beim Import (Trockenlauf und `--ohne-llm` laden den Judge nicht)
    from .stand import urteil_holen, urteil_merken
    from .tick import WIEDERHOLEN, _cache_antwort, _cache_eintrag

    e = einstellung(konfig)
    quelltext = quelltext or _quelltext_datei(konfig)
    zeit = jetzt or datetime.now().astimezone()
    schon = set(stand.erstgeprueft) if stand is not None else set()
    offen = sorted((p for p in posten if p.aussage.id not in schon),
                   key=lambda p: (p.aussage.datei, p.aussage.zeile_start, p.aussage.id))
    urteile: list[ErstUrteil] = []
    abbruch: str | None = None
    kw: dict[str, Any] = {"laufzeit": laufzeit, "agent": e.agent}
    if budget is not None:
        kw["budget"] = budget
    for p in offen:
        if not p.fragbar:
            continue
        a, b = p.aussage, p.beleg
        stand_digest = (digest(b) if digest is not None else None) or b.digest
        schl = (b.art, b.file, b.anker)
        treffer = urteil_holen(konfig, a.id, schl, stand_digest, stand_digest, fragetyp=FRAGETYP)
        if treffer is not None:
            antwort, grund = _cache_antwort(treffer, judge)
            urteile.append(ErstUrteil(p, antwort, grund, aus_cache=True))
            continue
        if deadline():
            abbruch = "deadline"
            break
        text = quelltext(b)
        frage = judge.JudgeFrage(a, b, text, text, nachbarn(b) if nachbarn is not None else (), fragetyp=FRAGETYP)
        antwort, grund = judge.zweimal_einig(frage, konfig, **kw)
        if antwort is None and grund.startswith(WIEDERHOLEN):
            abbruch = grund  # die Aussage bleibt offen; der naechste Lauf fragt beide Aufrufe neu
            break
        urteil_merken(konfig, a.id, schl, stand_digest, stand_digest, _cache_eintrag(antwort, grund, zeit),
                      fragetyp=FRAGETYP)
        urteile.append(ErstUrteil(p, antwort, grund))
    rest = len(offen) - len(urteile)
    return ErstErgebnis(len(urteile), rest, vollstaendig and rest == 0 and offene_seiten <= 0, tuple(urteile),
                        abbruch)


def stand_fortschreiben(stand: Stand, ergebnis: ErstErgebnis, *, dazu: Iterable[str] = ()) -> Stand:
    """Neuer Stand mit den Ids der Urteile aus `ergebnis` (und `dazu`) in `erstgeprueft` (sortiert) und `fertig`.
    Der Tick nutzt das nicht direkt: dort zaehlt eine Ersetzung erst nach ihrem Commit (`_Lauf.stand_speichern`)."""
    ids = set(stand.erstgeprueft) | {u.posten.aussage.id for u in ergebnis.urteile} | set(dazu)
    return dataclasses.replace(stand, erstgeprueft=sorted(ids),
                               erstbefuellung_fertig=stand.erstbefuellung_fertig or ergebnis.fertig)
