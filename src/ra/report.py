"""F11 report markdown in italiano (costruito da strutture verificate, non da testo libero dell'LLM)."""
def ref(d):
    link = f"https://arxiv.org/abs/{d['arxiv_id']}" if d.get("arxiv_id") else (f"https://doi.org/{d['doi']}" if d.get("doi") else "")
    au = ", ".join(d["authors"][:3]) + (" et al." if len(d["authors"]) > 3 else "")
    return f"- **[{d['key']}]** {au} ({d['year']}). *{d['title']}*. {d.get('venue') or ''} — {d['citations']} cit. {link}"

def L(items, num=False): return "\n".join((f"{i}. {x}" if num else f"- {x}") for i, x in enumerate(items, 1))

def render(r):
    sp, nv, gp, dirs, pl, corpus = r["spec"], r["novelty"], r["gaps"], r["directions"], r["plan"], r["corpus"]
    ch = r["chosen"]
    o = [f"# Piano di ricerca\n\n> **Idea:** {r['idea'].strip()}\n", f"*Run {r['run_id']} · {len(corpus)} paper nel corpus · citazioni valide {r['guard']['ok_pct']}%*\n"]
    if sp["clarifying_questions"]: o += ["## Domande di chiarimento\n", L(sp["clarifying_questions"]), "\n(Il piano procede con le assunzioni più ragionevoli.)\n"]
    o += ["## 1. Cosa investigare\n", f"**Problema:** {sp['problem']}\n\n**Metodo proposto:** {sp['proposed_method']}\n",
          f"### Novità: **{nv['verdict'].replace('_', ' ')}** (confidenza {nv.get('confidence') or 'n/d'}; {nv.get('close_papers', '?')} paper molto vicini trovati)\n\n{nv['summary']}\n", "**Confronto con i lavori più vicini**\n"]
    o += [f"- [{c['key']}] già fatto: {c['gia_fatto']} → *differenza:* {c['differenza']}" for c in nv["comparisons"]]
    o += ["\n**Evidenze testuali**\n"] + [f"- [{e['key']}] «{e['quote']}» {'✔' if e['verified'] else '(non verificata)'}" for e in nv["evidence"]]
    o += ["\n### Mappa dei gap\n"] + [f"- **{g['type']}** — {g['statement']} ({', '.join('['+k+']' for k in g['evidence_keys'])}). *Perché conta:* {g['why_it_matters']}" for g in gp]
    o += ["\n### Direzioni (ordinate per punteggio)\n", "| # | Direzione | Novità | Impatto | Fattib. | Rischio↓ | Profilo | **Totale** |", "|---|---|---|---|---|---|---|---|"]
    for i, d in enumerate(dirs, 1):
        s = d["scores"]; flag = " ⛔" if d.get("fatal_flaw") else ""
        o.append(f"| {i} | {d['title']}{flag} | {s['novelty']} | {s['impact']} | {s['feasibility']} | {s['risk_low']} | {s['profile_fit']} | **{d['total']}** |")
    for i, d in enumerate(dirs, 1):
        o += [f"\n**D{i}. {d['title']}** — {d['research_question']}\n", f"- Contributo: {d['expected_contribution']}", f"- Perché ora: {d['why_now']}",
              f"- Rischio: {d['main_risk']}", f"- Motivazione punteggio: {d['scores']['rationale']}"]
        c = d.get("critique")
        if c:
            o += [f"- Revisore — so what: {c['so_what']}", "- Debolezze: " + "; ".join(c["weaknesses"][:4]), "- Esperimenti che chiederebbe: " + "; ".join(c["reviewer_requests"][:4])]
            if c.get("fatal_flaw"): o.append(f"- ⛔ Difetto fatale: {c['fatal_reason']}")
            if c.get("revised_note"): o.append(f"- Lavoro vicino non visto: {c['revised_note']} (novità rivista a {c['revised_novelty']})")
    A, B, C = pl["A"], pl["B"], pl["C"]
    o += [f"\n## 2. Come svolgere la ricerca — direzione scelta: {ch['title']}\n", "### Domande di ricerca\n", L(A["research_questions"], True),
          "\n### Ipotesi falsificabili\n"] + [f"- **{h['id']}** {h['statement']} — metrica: {h['metric']}, soglia: {h['threshold']}" for h in A["hypotheses"]]
    o += ["\n### Contributi attesi\n", L(A["contributions"]), "\n### Posizionamento\n", L(A["positioning"]), f"\n### Metodo\n\n{A['method_overview']}\n", "**Scelte progettuali**\n", L(A["design_choices"]),
          "\n### Setup sperimentale\n", "**Baseline:** " + "; ".join(B["baselines"]), "\n**Dataset/benchmark:** " + "; ".join(B["datasets"]),
          "\n**Metriche primarie:** " + "; ".join(B["metrics_primary"]) + "\n\n**Secondarie:** " + "; ".join(B["metrics_secondary"]),
          "\n### Matrice degli esperimenti\n", "| ID | Ipotesi | Obiettivo | Procedura | Baseline | Metriche | Hardware | Giorni |", "|---|---|---|---|---|---|---|---|"]
    o += [f"| {e['id']} | {e['hypothesis']} | {e['goal']} | {e['procedure']} | {', '.join(e['baselines'])} | {', '.join(e['metrics'])} | {e['hardware']} | {e['estimated_days']} |" for e in B["experiments"]]
    o += ["\n**Ablation:** " + "; ".join(B["ablations"]), f"\n### Protocollo statistico\n\n{B['statistical_protocol']}\n", "### Minacce alla validità\n", L(B["threats"]),
          "\n### Rischi e piano B\n"] + [f"- **{x['risk']}** — segnale: {x['early_signal']} → piano B: {x['plan_b']}" for x in C["risks"]]
    o += ["\n### Venue target\n", L(C["venues"]), "\n### Scheletro del paper\n", L(C["paper_skeleton"]), "\n**Titoli:** " + " / ".join(C["title_options"]), f"\n**Abstract:** {C['abstract_pattern']}",
          "\n### Timeline\n", "| Settimane | Milestone | Criterio di uscita |", "|---|---|---|"] + [f"| {w['weeks']} | {w['milestone']} | {w['exit_criterion']} |" for w in C["timeline"]]
    fz = pl.get("feasibility")
    if fz:
        o += [f"\n### Carico stimato (calcolo, non opinione)\n", f"- Esperimenti: {fz['experiment_days']} giorni di lavoro ≈ {fz['hours']} ore ≈ **{fz['weeks_experiments']} settimane** a {fz['hours_per_week']} h/settimana; con 4 settimane di scrittura: **circa {fz['weeks_needed']} settimane**.",
              f"- Timeline del piano: {fz['timeline_weeks']} settimane" + (f" · settimane disponibili: {fz['available_weeks']}" if fz.get("available_weeks") else "")] + [f"- ⚠ {x}" for x in fz["flags"]]
    rv = pl.get("review")
    if rv:
        o += [f"\n### Controllo di coerenza del piano — verdetto: **{rv['verdict'].replace('_', ' ')}**{' (dopo 1 revisione automatica; prima: ' + pl['review_first']['verdict'].replace('_', ' ') + ', ' + str(sum(1 for i in pl['review_first']['issues'] if i['severity'] == 'alta')) + ' difetti gravi)' if rv.get('revised') else ''}\n"] + [f"- **{i['severity']} · {i['area']}** — {i['text']}" for i in rv["issues"]]
    o += [f"\n### ▶ Primo passo (domattina)\n\n{C['first_step']}\n", "## Pratica del campo (dal corpus)\n"] + [f"- **{k}:** {', '.join(v)}" for k, v in pl["practice"].items()]
    cited = sorted(corpus, key=lambda d: int(d["key"][1:]))
    o += ["\n## Bibliografia del corpus\n", "\n".join(ref(d) for d in cited), f"\n---\n*Controllo citazioni: {r['guard']['total']} riferimenti, {r['guard']['invalid']} non validi rimossi; citazioni testuali verificate {r['guard']['quotes_ok']}/{r['guard']['quotes']}.*\n"]
    return "\n".join(o)
