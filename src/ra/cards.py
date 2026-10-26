"""Schede strutturate per paper (F3). Livello 'full' (full text) o 'abstract' (solo abstract)."""
from typing import List, Optional
from pydantic import BaseModel, Field
from . import db, llm
from .config import CFG
from .parse import parse

class Card(BaseModel):
    problem: str = ""
    key_idea: str = ""
    contribution_type: str = Field("", description="method|system|benchmark|dataset|theory|survey|empirical-study")
    method_family: str = ""
    assumptions: List[str] = []
    baselines: List[str] = []
    datasets: List[str] = []
    metrics: List[str] = []
    hardware_scale: str = ""
    main_result: str = ""
    ablations: List[str] = []
    limitations: List[str] = []
    future_work: List[str] = []
    threats_to_validity: List[str] = []
    paper_structure: List[str] = []
    title_pattern: str = ""

SYS = "Sei un assistente che estrae dati strutturati da paper scientifici. Usa SOLO ciò che è nel testo; se manca, lascia vuoto. Rispondi in inglese tecnico, voci brevi."

def make_card(paper, full=True):
    level = "abstract"
    ctx = f"TITLE: {paper['title']}\nYEAR: {paper.get('year')}\nVENUE: {paper.get('venue','')}\nABSTRACT: {paper.get('abstract','')}"
    if full:
        d = parse(paper)
        if d and d["sections"]:
            s = d["sections"]
            part = lambda k, n: f"\n\n[{k.upper()}]\n{s[k][:n]}" if k in s else ""
            ctx += part("intro", 3500) + part("method", 3500) + part("experiments", 3500) + part("results", 2500) + part("discussion", 1500) + part("conclusion", 2000)
            if "method" not in s and "experiments" not in s: ctx += "\n\n[BODY EXCERPT]\n" + d.get("middle", "")
            if d["gap_sentences"]: ctx += "\n\n[GAP SENTENCES]\n" + "\n".join(d["gap_sentences"][:10])
            ctx += "\n\n[SECTION HEADINGS FOUND]\n" + ", ".join(s.keys())
            level = "full"
    c = llm.complete(f"Estrai la scheda strutturata di questo paper.\n\n{ctx}", Card, model=CFG["llm"]["extract" if level == "full" else "extract_fast"], system=SYS)
    out = c.model_dump(); out["level"] = level
    db.put_card(paper["id"], level, out)
    return out

def ensure_card(paper, full=True):
    c = db.get_card(paper["id"], "full" if full else None) or (None if full else db.get_card(paper["id"]))
    if c: return c
    try: return make_card(paper, full)
    except Exception as e:
        print(f"[card] {paper['id']}: {e}"); return None
