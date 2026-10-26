"""F12 valutazione: (cutoff) idee con data di taglio; (ld) idee iniziali dei progetti LD. Le metriche sono calcolate dal sistema e salvate in eval/results_*.json."""
import json, os, re
from typing import List
from pydantic import BaseModel
from . import llm
from .config import CFG, p
from .run import run_idea

class Hit(BaseModel):
    match: List[int]       # per ogni direzione, 1 se coincide con l'idea reale pubblicata
    baselines_found: int
    baselines_total: int

def judge(truth, ds, plan):
    sd = "\n".join(f"({i}) {d['title']}: {d['research_question']}" for i, d in enumerate(ds))
    bl = "; ".join(plan["B"]["baselines"])
    return llm.complete(f"""Giudice imparziale. Ecco la ricerca REALE pubblicata: {truth['title']} — {truth['summary']}. Baseline realmente usate: {truth['baselines']}.
DIREZIONI PROPOSTE dal sistema:
{sd}
BASELINE PROPOSTE: {bl}
match: per ogni direzione 1 se affronta sostanzialmente la stessa domanda/contributo della ricerca reale, altrimenti 0. baselines_found = quante baseline reali compaiono (anche con nome diverso) tra quelle proposte; baselines_total = numero baseline reali.""", Hit, model=CFG["llm"]["reason"])

def corpus_recall(case, corpus):
    """metrica oggettiva: i paper delle baseline reali sono stati recuperati nel corpus? (match sul titolo)"""
    bt = case["truth"].get("baseline_titles") or []
    if not bt: return None
    titles = " | ".join(d["title"].lower() for d in corpus)
    return round(sum(1 for t in bt if t.lower() in titles) / len(bt), 2)

def evaluate(which, n=5):
    cases = json.load(open(p("eval", f"cases_{which}.json")))[:n]; res = []
    for c in cases:
        r = run_idea(c["idea"], cutoff=c.get("cutoff"), fast=True, run_id=f"eval-{which}-{c['id']}", log=lambda *_: None)
        j = judge(c["truth"], r["directions"], r["plan"])
        ranks = [i + 1 for i, m in enumerate(j.match) if m]
        res.append({"id": c["id"], "hit_at_5": int(any(x <= 5 for x in ranks)), "first_hit_rank": ranks[0] if ranks else None,
                    "baseline_recall": (round(j.baselines_found / j.baselines_total, 2) if c["truth"]["baselines"] and j.baselines_total else None), "citations_ok_pct": r["guard"]["ok_pct"], "corpus_recall": corpus_recall(c, r["corpus"]),
                    "quotes_verified": f"{r['guard']['quotes_ok']}/{r['guard']['quotes']}"})
        print(res[-1])
    summ = {"n": len(res), "hit_at_5": round(sum(x["hit_at_5"] for x in res) / len(res), 2),
            "baseline_recall": (round(sum(x["baseline_recall"] for x in res if x["baseline_recall"] is not None) / sum(1 for x in res if x["baseline_recall"] is not None), 2) if any(x["baseline_recall"] is not None for x in res) else None),
            "corpus_recall": (round(sum(x["corpus_recall"] for x in res if x["corpus_recall"] is not None) / sum(1 for x in res if x["corpus_recall"] is not None), 2) if any(x["corpus_recall"] is not None for x in res) else None),
            "citations_ok_pct": round(sum(x["citations_ok_pct"] for x in res) / len(res), 1)}
    json.dump({"summary": summ, "cases": res}, open(p("eval", f"results_{which}.json"), "w"), indent=1, ensure_ascii=False)
    print(summ)


def eval_retrieval(n=6, which="cutoff"):
    """valutazione veloce solo del recupero (F5-F6): le baseline reali sono nel pool candidati e nel corpus finale?"""
    from . import retrieve as R
    cases = json.load(open(p("eval", f"cases_{which}.json")))[:n]; out = []
    for c in cases:
        spec = R.analyse_idea(c["idea"]); pool = R.collect(spec, c.get("cutoff"), lambda *_: None)
        corpus = R.rank(spec, pool, lambda *_: None)
        out.append({"id": c["id"], "pool": len(pool), "corpus": len(corpus), "pool_recall": corpus_recall(c, pool), "corpus_recall": corpus_recall(c, corpus)})
        print(out[-1])
    ok = [x for x in out if x["corpus_recall"] is not None]
    sm = {"n": len(out), "pool_recall": round(sum(x["pool_recall"] for x in ok) / max(1, len(ok)), 2), "corpus_recall": round(sum(x["corpus_recall"] for x in ok) / max(1, len(ok)), 2)}
    json.dump({"summary": sm, "cases": out}, open(p("eval", "results_retrieval.json"), "w"), indent=1, ensure_ascii=False); print(sm)
