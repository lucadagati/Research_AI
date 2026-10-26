"""F7 novità + gap, F8 direzioni + punteggio, F9 revisore critico."""
import json
from typing import List, Optional
from pydantic import BaseModel
from . import llm, harvest as h, guard
from .retrieve import Rel, _before
from .config import CFG, RES

W = CFG["scoring_weights"]
IT = "IMPORTANTE: scrivi TUTTI i campi testuali in ITALIANO, anche milestone, primo passo, rischi e procedure (solo termini tecnici, nomi di metodi/dataset e titoli di paper in inglese). "

def brief(d, full=False):
    c = d.get("card") or {}
    s = f"[{d['key']}] {d['title']} ({d['year']}, {d['citations']} cit, {d.get('venue') or 'n/d'}, {'pilastro' if d['pillar'] else 'frontiera' if d['frontier'] else 'altro'})"
    if c:
        s += f"\n  problema: {c.get('problem','')[:200]} | metodo: {c.get('method_family','')} {c.get('key_idea','')[:150]}"
        if c.get("assumptions"): s += f"\n  assunzioni: {'; '.join(c['assumptions'][:3])[:250]}"
        if c.get("limitations") or c.get("future_work"): s += f"\n  limiti/futuro: {'; '.join((c.get('limitations') or [])[:3] + (c.get('future_work') or [])[:2])[:350]}"
        if c.get("datasets") or c.get("metrics"): s += f"\n  dataset: {', '.join(c.get('datasets',[])[:4])} | metriche: {', '.join(c.get('metrics',[])[:4])}"
    else: s += f"\n  abstract: {d['abstract'][:300]}"
    return s

# ---------- F7 novità ----------
class Comp(BaseModel):
    key: str
    gia_fatto: str
    differenza: str
class Quote(BaseModel):
    key: str
    quote: str
class Novelty(BaseModel):
    verdict: str   # gia_fatto | incrementale | differenza_chiara | nuovo
    confidence: str = ""   # alta | media | bassa (bassa se i paper trovati non sono davvero vicini)
    summary: str
    comparisons: List[Comp]
    evidence: List[Quote]

def novelty(spec, corpus, direction=None, stats=None):
    near = sorted(corpus, key=lambda d: (-d["rel"], -d.get("sim", 0)))[:6]
    close = sum(1 for d in near if d["rel"] >= 3)
    tgt = direction or f"{spec.problem} | metodo: {spec.proposed_method}"
    out = llm.complete(f"""{IT}Valuta la novità di questa proposta rispetto ai paper più vicini.
PROPOSTA: {tgt}
PAPER VICINI:
""" + "\n".join(brief(d) + f"\n  abstract: {d['abstract'][:1100]}" for d in near) + """

Verdetto: gia_fatto | incrementale | differenza_chiara | nuovo. Per ogni paper vicino: cosa fa già (gia_fatto) e in cosa la proposta differisce.
In evidence riporta citazioni TESTUALI di 10-25 parole COPIATE CARATTERE PER CARATTERE dall'abstract qui sopra (niente parafrasi, niente virgolette interne) a supporto, con la chiave [Pk] del paper. Sii severo: se è già stato fatto, dillo. Confronta solo i paper davvero vicini: se nessuno lo è (qui {close} su {len(near)} sono marcati molto vicini) scrivilo nel riepilogo e metti confidence=bassa; l'assenza di lavori trovati NON prova che l'idea sia nuova.""", Novelty)
    by = {d["key"]: d for d in corpus}
    ev = []
    for q in out.evidence:
        k = q.key.strip("[] ")
        ok = k in by and guard.quote_ok(q.quote, by[k])
        if stats is not None: stats["quotes"] += 1; stats["quotes_ok"] += int(ok)
        ev.append({"key": k, "quote": q.quote, "verified": ok})
    conf = out.confidence
    if close < 2 or not ev:   # regola dura: poche prove => la novità non è affermabile con alta confidenza
        conf = "bassa"
    return {"verdict": out.verdict, "confidence": conf, "close_papers": close, "summary": out.summary,
            "comparisons": [c.model_dump() for c in out.comparisons if c.key.strip("[] ") in by], "evidence": ev}

# ---------- F7 gap ----------
class Gap(BaseModel):
    type: str   # G1..G6
    statement: str
    evidence_keys: List[str]
    why_it_matters: str
class Gaps(BaseModel):
    gaps: List[Gap]

GAPDEF = """G1 dichiarato: limite/future work segnalato da più paper
G2 combinazione mancante: metodo x setting/dataset non coperto dai paper
G3 contraddizione: risultati o claim discordanti
G4 assunzione fragile: assunzione condivisa dai pilastri e messa in dubbio dalla frontiera
G5 lacuna di valutazione: manca benchmark standard, pochi run, assenza di test statistici o di baseline forti
G6 trasferimento: tecnica forte in un'area vicina non ancora applicata qui"""

def gaps(spec, corpus):
    out = llm.complete(f"""{IT}Mappa i gap di ricerca rilevanti per questa idea, usando SOLO i paper del corpus.
IDEA: {spec.problem} | metodo: {spec.proposed_method}
TIPI DI GAP:
{GAPDEF}
CORPUS ({len(corpus)} paper):
""" + "\n".join(brief(d) for d in corpus[:70]) + """

Produci 8-14 gap concreti e specifici (non generici). Ogni gap: tipo (G1..G6), enunciato, evidence_keys = almeno 2 chiavi [Pk] dei paper che lo sostengono
(per G1 conta quanti paper lo dicono), perché conta per l'idea.""", Gaps)
    return out.gaps

# ---------- F8 direzioni ----------
class Direction(BaseModel):
    title: str
    research_question: str
    expected_contribution: str
    why_now: str
    main_risk: str
    gap_indexes: List[int]
    evidence_keys: List[str]
class Dirs(BaseModel):
    directions: List[Direction]
class Score(BaseModel):
    novelty: int
    impact: int
    feasibility: int
    risk_low: int
    profile_fit: int
    rationale: str
class Scores(BaseModel):
    scores: List[Score]

def directions(spec, corpus, gp):
    g = "\n".join(f"({i}) [{x.type}] {x.statement}" for i, x in enumerate(gp))
    out = llm.complete(f"""{IT}Proponi 6-8 direzioni di ricerca DISTINTE che partono dall'idea e sfruttano i gap trovati.
IDEA: {spec.problem} | metodo: {spec.proposed_method}
GAP:
{g}
Per ogni direzione: titolo, domanda di ricerca, contributo atteso, perché ora, rischio principale, gap_indexes (indici dei gap che affronta), evidence_keys ([Pk] dal corpus: {', '.join(d['key'] for d in corpus[:60])}).
La PRIMA direzione deve essere l'idea dell'utente nella sua forma più solida e focalizzata (non una variante laterale). Le altre devono essere diverse tra loro: almeno una conservativa, una ambiziosa, una di valutazione/benchmark.""", Dirs)
    ds = out.directions
    res = f"{RES['hours_per_week']} ore/settimana; calcolo: {'; '.join(RES['compute'])}; board: {'; '.join(RES['boards'])}; testbed: {'; '.join(RES['testbeds'])}; NON disponibile: {'; '.join(RES['no_access'])}"
    sc = llm.complete(f"""{IT}Valuta ogni direzione con punteggi interi 1-5 (5=meglio) e una motivazione breve.
novelty: distanza dai lavori esistenti | impact: crescita e rilevanza del tema | feasibility: fattibile con QUESTE risorse: {res}
risk_low: 5 = rischio basso | profile_fit: coerenza con il profilo {', '.join(RES['profile']['topics'])}
Restituisci scores nello stesso ordine delle {len(ds)} direzioni.
""" + "\n".join(f"({i}) {d.title}: {d.research_question} | contributo: {d.expected_contribution} | rischio: {d.main_risk}" for i, d in enumerate(ds)), Scores)
    outl = []
    for d, s in zip(ds, sc.scores + [None] * len(ds)):
        if s is None: continue
        dd = d.model_dump(); dd["scores"] = s.model_dump(); dd["total"] = total(s); outl.append(dd)
    return sorted(outl, key=lambda x: -x["total"])

def total(s):
    g = lambda k: s[k] if isinstance(s, dict) else getattr(s, k)
    return round(sum(W[k] * g(f) for k, f in [("novelty", "novelty"), ("impact", "impact"), ("feasibility", "feasibility"),
                                              ("risk", "risk_low"), ("profile_fit", "profile_fit")]), 3)

# ---------- F9 revisore ----------
class Critique(BaseModel):
    weaknesses: List[str]
    so_what: str
    reviewer_requests: List[str]
    prior_work_queries: List[str]
    fatal_flaw: bool
    fatal_reason: str = ""
class Revised(BaseModel):
    novelty: int
    note: str

def critique(direction, corpus, spec, log=print, cutoff=None):
    c = llm.complete(f"""{IT}Sei un revisore severo di una rivista IEEE/Elsevier. Critica questa direzione di ricerca.
DIREZIONE: {direction['title']} — {direction['research_question']} | contributo: {direction['expected_contribution']}
Elenca debolezze metodologiche, il "so what?", gli esperimenti che chiederesti, 1-2 query (in inglese) per cercare lavori precedenti che potrebbero averla già fatta,
e dì se c'è un difetto fatale (fatal_flaw) con motivo.
Paper del corpus: """ + "\n".join(brief(d) for d in corpus[:12]), Critique)
    out = c.model_dump(); out["new_papers"] = []
    found = []
    for q in c.prior_work_queries[:2]:
        for d in h.arxiv_search(q, 6):
            if not _before(d, cutoff): continue
            if d["id"] not in {x["id"] for x in corpus} and d["id"] not in {x["id"] for x in found}:
                found.append(d)
    if found:
        rr = llm.complete(f"""{IT}Tra questi paper nuovi, quali sono davvero vicini a questa direzione ("{direction['title']}: {direction['research_question']}")?
Restituisci solo il JSON {{"scores":[...]}} con un intero 0-3 per paper nello stesso ordine.
""" + "\n".join(f"[{i}] {d['title']}: {d['abstract'][:300]}" for i, d in enumerate(found)), Rel, model=CFG["llm"]["extract_fast"])
        close = [d for d, s in zip(found, rr.scores) if s >= 2]
        if close:
            n0 = len(corpus)
            for i, d in enumerate(close, 1):
                h.label(d); d["rel"] = 2; d["sim"] = 0; d["card"] = None; d["key"] = f"P{n0 + i}"
            rv = llm.complete(f"""{IT}Dato il nuovo lavoro vicino trovato dal revisore, rivaluta la novità (1-5) della direzione "{direction['title']}".
""" + "\n".join(f"[{d['key']}] {d['title']}: {d['abstract'][:500]}" for d in close), Revised)
            out["new_papers"] = close
            out["revised_novelty"] = rv.novelty; out["revised_note"] = rv.note
            log(f"    revisore: {len(close)} lavori vicini non visti")
    return out
