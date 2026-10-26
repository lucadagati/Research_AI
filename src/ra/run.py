"""Orchestrazione end-to-end: ra idea "..." -> runs/<ts>/ (F5-F11)."""
import json, os, time
from . import cards, retrieve as R, analysis as A, plan as P, guard, report, harvest as h
from .config import CFG, p

def save(rd, name, obj):
    json.dump(obj, open(os.path.join(rd, name), "w"), ensure_ascii=False, indent=1, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o))

def _load(rd, name):
    f = os.path.join(rd, name)
    return json.load(open(f)) if os.path.exists(f) else None

def run_idea(idea, cutoff=None, venue=None, weeks=None, fast=False, direction=None, run_id=None, log=print):
    t0 = time.time(); rid = run_id or time.strftime("%Y%m%d-%H%M%S"); rd = p("runs", rid); os.makedirs(rd, exist_ok=True)
    open(os.path.join(rd, "idea.txt"), "w").write(idea)
    save(rd, "meta.json", {"cutoff": cutoff, "venue": venue, "weeks": weeks, "fast": fast, "llm": CFG["llm"], "weights": CFG["scoring_weights"]})
    stats = {"total": 0, "invalid": 0, "quotes": 0, "quotes_ok": 0}
    sj = _load(rd, "spec.json")   # ripresa: le fasi già salvate in runs/<id>/ non si rifanno
    if sj: spec = R.IdeaSpec(**sj); log("F5 (ripreso)")
    else: log("F5 analisi idea"); spec = R.analyse_idea(idea, venue); save(rd, "spec.json", spec)
    corpus = _load(rd, "corpus.json")
    if corpus: log("F6/F3 (ripreso)")
    else:
        log("F6 recupero"); pool = R.collect(spec, cutoff, log); corpus = R.rank(spec, pool, log)
        log("F3 schede"); corpus = R.build_cards(corpus, fast, log); save(rd, "corpus.json", corpus)
    valid = {d["key"] for d in corpus}
    nv = _load(rd, "novelty.json")
    if not nv: log("F7 novità"); nv = A.novelty(spec, corpus, stats=stats); save(rd, "novelty.json", nv)
    gj = _load(rd, "gaps.json")
    if gj: gp = [A.Gap(**g) for g in gj]
    else:
        log("F7 gap"); gp = A.gaps(spec, corpus)
        for g in gp: g.evidence_keys = guard.valid_ids([k.strip("[] ") for k in g.evidence_keys], valid, stats)
        gp = [g for g in gp if len(g.evidence_keys) >= 2]; save(rd, "gaps.json", gp)
    log(f"   {len(gp)} gap con evidenze")
    ds = _load(rd, "directions.json")
    if ds: log("F8/F9 (ripreso)")
    else: ds = _directions(spec, corpus, gp, stats, valid, log, cutoff); save(rd, "directions.json", ds); save(rd, "corpus.json", corpus)
    chosen = ds[direction - 1] if direction else next((d for d in ds if not d.get("fatal_flaw")), ds[0])
    log(f"F10 piano per: {chosen['title']}")
    pl = _load(rd, "plan_raw.json")
    if not pl: pl = P.make_plan(spec, chosen, corpus, gp, venue, weeks, cutoff, idea); save(rd, "plan_raw.json", pl)
    return _finish(rd, rid, idea, spec, corpus, nv, gp, ds, pl, chosen, stats, t0, log, weeks)

def _directions(spec, corpus, gp, stats, valid, log, cutoff=None):
    log("F8 direzioni"); ds = A.directions(spec, corpus, gp)
    for d in ds: d["evidence_keys"] = guard.valid_ids([k.strip("[] ") for k in d["evidence_keys"]], valid, stats)
    log("F9 revisore")
    for i, d in enumerate(ds[:4]):
        c = A.critique(d, corpus, spec, log, cutoff)
        for np_ in c.pop("new_papers"):
            corpus.append(np_); valid.add(np_["key"]); db_put(np_); np_["card"] = cards.ensure_card(np_, full=False)
        d["critique"] = c; d["fatal_flaw"] = c["fatal_flaw"]
        if "revised_novelty" in c:
            d["scores"]["novelty"] = min(d["scores"]["novelty"], c["revised_novelty"]); d["total"] = A.total(d["scores"])
        if c["fatal_flaw"]: d["total"] = round(d["total"] - 1, 3)
    ds.sort(key=lambda x: -x["total"])
    return ds

def _finish(rd, rid, idea, spec, corpus, nv, gp, ds, pl, chosen, stats, t0, log, meta_weeks=None):
    valid = {d["key"] for d in corpus}
    # cite guard su tutto l'output testuale
    pl["feasibility"] = P.feasibility(pl, meta_weeks)
    pl = P.italianize(pl)
    pl = guard.clean(pl, valid, stats); nv = guard.clean(nv, valid, stats)
    gpd = guard.clean([g.model_dump() for g in gp], valid, stats); ds = guard.clean(ds, valid, stats)
    from . import llm as _llm
    stats["ctx_truncations"] = len(_llm.WARNINGS)
    stats["ok_pct"] = round(100 * (1 - stats["invalid"] / max(1, stats["total"])), 1); save(rd, "guard.json", stats)
    res = {"idea": idea, "run_id": rid, "spec": spec.model_dump(), "novelty": nv, "gaps": gpd, "directions": ds, "plan": pl,
           "corpus": [{k: v for k, v in d.items() if k != "card"} for d in corpus], "chosen": chosen, "guard": stats}
    save(rd, "plan.json", pl)
    open(os.path.join(rd, "report.md"), "w").write(report.render(res))
    log(f"Fatto in {round(time.time()-t0)}s → runs/{rid}/report.md")
    return res

def db_put(d):
    from . import db; db.put_paper({k: v for k, v in d.items() if k != "card"})
