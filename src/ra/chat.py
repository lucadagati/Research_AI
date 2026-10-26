"""F11 chat di approfondimento su un run (RAG sul corpus del run)."""
import json, os
import numpy as np
from . import llm
import yaml
from .config import CFG, RES, p

def chat(run):
    rd = p("runs", run); corpus = json.load(open(f"{rd}/corpus.json")); report = open(f"{rd}/report.md").read()
    V = llm.embed([d["title"] + ". " + d["abstract"] for d in corpus]); hist = []
    print("Chat sul run", run, "(esc con Ctrl-D)")
    while True:
        try: q = input("\n> ")
        except EOFError: break
        qv = llm.embed([q], "query")[0]; top = [corpus[i] for i in np.argsort(-(V @ qv))[:8]]
        ctx = "\n".join(f"[{d['key']}] {d['title']} ({d['year']}): {d['abstract'][:500]}" for d in top)
        r = llm.chat(CFG["llm"]["reason"], f"RISORSE REALI DELL'AUTORE (usa solo queste; 'Coral' = Coral Edge TPU, NON uno strumento software): {yaml.safe_dump(RES, allow_unicode=True)[:1500]}\n\nREPORT DEL RUN:\n{report[:12000]}\n\nPAPER PERTINENTI:\n{ctx}\n\nCONVERSAZIONE:\n{''.join(hist[-6:])}\nDOMANDA: {q}\n"
                     "Rispondi in italiano, conciso, citando solo chiavi [Pk] esistenti. Se ti chiedono di rifare il piano con altri vincoli, riscrivi le parti cambiate.")
        print(r); hist.append(f"D: {q}\nR: {r}\n")
