"""F5 analisi idea + F6 recupero su misura."""
import re
from typing import List
import numpy as np
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from . import db, llm, harvest as h, cards
from .config import CFG, RES

R = CFG["retrieval"]

class IdeaSpec(BaseModel):
    problem: str
    domain: str
    proposed_method: str
    implicit_assumptions: List[str] = []
    expected_contribution_type: str = ""
    keywords: List[str] = []
    search_queries: List[str] = []
    risky_claims: List[str] = []
    clarifying_questions: List[str] = []

def analyse_idea(idea, venue=None):
    ctx = f"\nVenue target indicato: {venue}" if venue else ""
    return llm.complete(f"""Analizza questa idea di ricerca di un ricercatore.
REGOLE: non aggiungere metodi, domini applicativi o tecnologie che l'idea NON menziona (il profilo dell'autore serve solo più avanti, non per riempire i vuoti). Se l'idea è vaga, formula problema e metodo in modo AMPIO e fedele al testo, e metti in clarifying_questions le 2-3 scelte che andrebbero chiarite (scopo, carichi/dominio, vincoli, metriche).
IDEA:
{idea}{ctx}

Compila: problema, dominio, metodo proposto, assunzioni implicite, tipo di contributo atteso, parole chiave,
{R['queries_per_idea']} query di ricerca bibliografica IN INGLESE (brevi, 2-4 parole). Servono: almeno 3 query che COMBINANO due concetti dell'idea
(es. metodo + dominio, metodo + garanzia), una sul problema da solo, una sul metodo da solo, una su benchmark/valutazione, alcune con sinonimi.
Evita query troppo generiche (una sola parola comune) e termini inventati, affermazioni rischiose (claim che un revisore metterebbe in dubbio),
e 0-3 domande di chiarimento solo se l'idea è davvero ambigua. Testo in italiano, query in inglese.""", IdeaSpec)

class RelItem(BaseModel):
    i: int
    score: int
class Rel(BaseModel):
    scores: List[int] = []
class RelOut(BaseModel):
    items: List[RelItem]

def _before(d, cutoff):
    dt = d.get("date") or (f"{d['year']}-12-31" if d.get("year") else None)
    return (not cutoff) or (dt is not None and dt < cutoff)

def collect(spec, cutoff=None, log=print):
    pool = []
    qs = [" ".join(re.findall(r"[\w-]+", q)) for q in spec.search_queries[:R["queries_per_idea"]]]
    for i, q in enumerate(qs):
        a = h.arxiv_search(q, R["arxiv_per_query"]); pool += a
        s = h.s2_search(q, R["s2_per_query"]); pool += s
        if i < 4: pool += h.ieee_oa(q, pages=1, min_cit=20, prefix=None, rank="relevance", from_year=CFG["from_year"])
        log(f"  query '{q}': arxiv={len(a)} s2={len(s)}")
    # arricchisci con citazioni S2 i record senza conteggio (arXiv)
    need = [d for d in pool if d["source"] == "arxiv" and d["id"].startswith("arxiv:")]
    ids = list(dict.fromkeys("ARXIV:" + d["arxiv_id"] for d in need))
    info = dict(zip(ids, h.s2_batch(ids))) if ids else {}
    for d in need:
        r = info.get("ARXIV:" + d["arxiv_id"])
        if r:
            d["citations"], d["influential"], d["s2_id"] = r["citations"], r["influential"], r["s2_id"]
            d["venue"] = d["venue"] or r["venue"]; d["doi"] = d["doi"] or r["doi"]
    # base locale
    ids_l, M = db.get_vecs("doc")
    if len(ids_l):
        qv = llm.embed([spec.problem + ". " + spec.proposed_method], "query")[0]
        for j in np.argsort(-(M @ qv))[:40]:
            pp = db.get_paper(ids_l[j]); pool.append(pp) if pp else None
    pool = [d for d in h.merge(pool) if d.get("abstract") and _before(d, cutoff)]
    pool += expand(spec, pool, cutoff, log)          # pilastri dai riferimenti dei lavori più vicini
    pool = [d for d in h.merge(pool) if d.get("abstract") and _before(d, cutoff)]
    for d in pool: h.label(d)
    h.frontier_rank(pool)
    log(f"  candidati unici con abstract: {len(pool)}")
    return pool

def expand(spec, pool, cutoff, log=print, seeds=8):
    """espansione nel grafo: riferimenti molto citati dei `seeds` paper più vicini (per similarità) -> pilastri del campo"""
    if len(pool) < 3: return []
    qv = llm.embed([spec.problem + ". " + spec.proposed_method], "query")[0]
    V = llm.embed([d["title"] + ". " + d["abstract"] for d in pool])
    out = []
    for i in np.argsort(-(V @ qv))[:seeds]:
        for x in h.s2_neighbors(pool[i], "references", 100):
            if x["citations"] >= CFG["niche_min_citations"] and x.get("abstract") and _before(x, cutoff): out.append(x)
    log(f"  espansione riferimenti: +{len(out)} candidati molto citati")
    return out

def rank(spec, pool, log=print):
    qv = llm.embed([spec.problem + ". " + spec.proposed_method + ". " + " ".join(spec.keywords)], "query")[0]
    V = llm.embed([d["title"] + ". " + d["abstract"] for d in pool])
    sim = V @ qv
    order = np.argsort(-sim)[:R["pool"]]
    cand = [pool[i] for i in order]
    for i, d in zip(order, cand): d["sim"] = round(float(sim[i]), 4)
    # rilevanza LLM a blocchi
    def score(block):
        txt = "\n".join(f"[{i}] {d['title']} ({d['year']}): {d['abstract'][:400]}" for i, d in enumerate(block))
        try:
            r = llm.complete(f"""IDEA: problema = {spec.problem} | metodo = {spec.proposed_method}
Sei un bibliotecario severo. Per OGNI paper dai un punteggio di rilevanza per questa idea:
3 = affronta lo stesso problema OPPURE usa lo stesso metodo nello stesso dominio applicativo (lavoro vicino, da citare e da cui differenziarsi)
2 = stesso metodo in un altro dominio, oppure stesso dominio con un metodo diverso ma comparabile (utile come baseline/related work)
1 = tema generico o solo una parola in comune
0 = irrilevante (dominio e metodo diversi, es. immagini, medicina, pura teoria non collegata)
Nel dubbio dai il punteggio più basso. Restituisci items = [{{i, score}}] per ciascuno dei {len(block)} paper.
{txt}""", RelOut, model=CFG["llm"]["reason"])
            sc = {x.i: x.score for x in r.items}
            return [max(0, min(3, sc.get(i, 0))) for i in range(len(block))]
        except Exception as e:
            print("[rel]", e); return [1] * len(block)
    blocks = [cand[i:i + 10] for i in range(0, len(cand), 10)]
    with ThreadPoolExecutor(CFG["llm"]["parallel"]) as ex: res = list(ex.map(score, blocks))
    for b, sc in zip(blocks, res):
        for d, s in zip(b, sc): d["rel"] = s
    keep = [d for d in cand if d["rel"] >= R["relevance_min"]]
    # bilanciamento pilastri/frontiera
    n = max(len(keep), 1)
    def add(pred, share):
        nonlocal keep
        have = sum(1 for d in keep if pred(d))
        extra = [d for d in cand if d not in keep and d["rel"] >= 1 and pred(d)]
        need = int(share * n) - have
        keep += extra[:max(0, need)]
    add(lambda d: d["pillar"], R["min_pillar_share"]); add(lambda d: d["frontier"], R["min_frontier_share"])
    # corsia "canonici": metodi generali molto citati (es. PPO, DDPG) sono spesso le baseline ma hanno rilevanza 1
    near = {id(d) for d in cand[:60]}     # i 60 più vicini per embedding
    canon = sorted([d for d in cand if d not in keep and (d["rel"] >= 1 or id(d) in near) and d["citations"] >= R.get("canonical_min_citations", 500)],
                   key=lambda d: -d["citations"])[:R.get("canonical_max", 10)]
    for d in canon: d["lane"] = "canonical"
    keep += canon
    keep.sort(key=lambda d: (d.get("lane") == "canonical", -d["rel"], -d["sim"]))
    for i, d in enumerate(keep, 1): d["key"] = f"P{i}"
    log(f"  corpus: {len(keep)} paper (pilastri {sum(d['pillar'] for d in keep)}, frontiera {sum(d['frontier'] for d in keep)})")
    return keep

def build_cards(corpus, fast=False, log=print):
    for d in corpus: db.put_paper(d)
    order = sorted(corpus, key=lambda d: (-d["rel"], -d["sim"]))
    full_ids = {d["id"] for d in order if d.get("pdf_url")}
    full_ids = set([d["id"] for d in order if d["id"] in full_ids][:0 if fast else R["full_cards"]])
    def job(d):
        return d["id"], cards.ensure_card(d, full=d["id"] in full_ids)
    with ThreadPoolExecutor(CFG["llm"]["parallel"]) as ex: res = dict(ex.map(job, corpus))
    for d in corpus: d["card"] = res.get(d["id"])
    log(f"  schede: {sum(1 for d in corpus if d['card'])}/{len(corpus)} (full: {sum(1 for d in corpus if d['card'] and d['card'].get('level')=='full')})")
    return corpus
