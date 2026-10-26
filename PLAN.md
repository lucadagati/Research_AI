# Research_AI — Piano di sviluppo

> **Obiettivo.** Dai un'idea di ricerca in input. Il sistema restituisce:
> **(A)** cosa conviene investigare (direzioni con punteggio, controllo di novità e gap);
> **(B)** come svolgere l'intera ricerca (domande di ricerca, ipotesi, metodo, baseline, dataset,
> esperimenti, analisi statistica, rischi, venue, scheletro del paper, timeline).
> Ogni affermazione è ancorata a paper reali e open (arXiv, IEEE OA, altri OA).

---

## 0. Principi e vincoli

| # | Principio | Conseguenza pratica |
|---|---|---|
| P1 | **Solo fonti aperte, nessuna API key** | arXiv API, Semantic Scholar anonimo, Crossref, Unpaywall (solo email), OpenCitations come riserva |
| P2 | **Nessuna citazione inventata** | l'LLM cita solo ID presenti nel corpus; un validatore scarta gli ID sconosciuti |
| P3 | **Pilastri + frontiera** | pilastri = ≥100 citazioni (normalizzate per età) → *come* si fa ricerca; frontiera = ultimi 24 mesi, citazioni in crescita → *cosa* è aperto |
| P4 | **Fattibilità reale** | il piano conosce le risorse disponibili (L40, rpi4, vf2, Coral, testbed LoRaWAN, budget di tempo) |
| P5 | **Tutto locale** | LLM su Ollama (L40); un'API per il ragionamento è opzionale e disattivata di default |
| P6 | **Riproducibile** | ogni run salva query, ID recuperati, prompt, versione del modello e output in `runs/<timestamp>/` |
| P8 | **Max 50% della GPU** | tetto sulla VRAM dei miei modelli (`llm.gpu_budget_frac`), un solo run alla volta (coda), modelli scaricati dopo 3 min di inattività |
| P7 | **Valutabile** | test con data di taglio e test sui progetti LD esistenti prima di considerarlo "finito" |

**Risorse verificate (2026-10-17):**
- L40, 46 GB, condivisa; Python 3.10; 4.4 TB liberi.
- Ollama con `gpt-oss:20b`, `mistral-small:24b`, `qwen2.5:14b`, `phi4:14b`, `gemma3:12b`, `nomic-embed-text`, `qwen2.5vl:32b`.

---

## 1. Architettura

```
                ┌─────────────────────────── OFFLINE (una volta + aggiornamento settimanale) ─────────────┐
 config.yaml ──►│ F1 Harvest ─► F2 Parse PDF ─► F3 Schede strutturate (LLM) ─► F4 Indice (vettori + grafo) │
                └──────────────────────────────────────────────────────────────────────────┬──────────────┘
                                                                                           │ base locale
                ┌─────────────────────────── ONLINE (per ogni idea) ───────────────────────▼──────────────┐
 "la mia idea" ►│ F5 Analisi idea ─► F6 Recupero su misura ─► F7 Novità + gap ─► F8 Direzioni + punteggio │
                │        ─► F9 Revisore critico ─► F10 Piano di ricerca completo ─► F11 Report + chat      │
                └──────────────────────────────────────────────────────────────────────────────────────────┘
                F12 Valutazione (test con data di taglio, test su LD1–LD11)
```

### Struttura del repo
```
Research_AI/
├── PLAN.md  README.md  config.yaml  resources.yaml   # resources.yaml = le tue risorse (P4)
├── src/ra/
│   ├── harvest/      arxiv.py  s2.py  crossref.py  unpaywall.py  ieee_oa.py
│   ├── parse/        pdf.py  sections.py
│   ├── extract/      card.py  prompts/
│   ├── index/        store.py  graph.py  embed.py
│   ├── pipeline/     idea.py  retrieve.py  novelty.py  gaps.py  directions.py  critic.py  plan.py  report.py
│   ├── llm.py        # wrapper unico: Ollama (default) | API (opzionale)
│   ├── cite_guard.py # validatore citazioni (P2)
│   └── cli.py        # `ra ingest`, `ra idea "..."`, `ra chat`, `ra eval`
├── data/             # gitignored: raw/, pdf/, text/, cards/, db.sqlite, vectors/
├── runs/             # gitignored: un run per idea
└── eval/             # casi di test + metriche
```

---

## 2. Fasi

Formato per ogni fase: **Scopo · Input · Passi · Output · Criterio di accettazione.**

### F0 — Setup (½ giorno)
- **Scopo:** ambiente pulito e riproducibile.
- **Passi:**
  1. `python -m venv .venv`; dipendenze: `pyyaml requests pymupdf lancedb networkx pydantic typer rich tenacity`.
  2. Spostare gli script attuali in `src/ra/harvest/` (`fetch_arxiv.py` → `arxiv.py` + `s2.py`; `fetch_ieee_oa.py` → `crossref.py` + `unpaywall.py`).
  3. Scrivere `resources.yaml`: hardware, testbed, dataset già disponibili, ore/settimana, scadenze. Da `info_testbed` e dalla memoria dei progetti.
  4. `llm.py`: `complete(prompt, schema)` con output JSON validato da pydantic e retry se il JSON è malformato.
  5. Database SQLite con tabelle `papers`, `citations`, `cards`, `runs`.
- **Accettazione:** `ra --help` funziona; `llm.complete` restituisce JSON valido con `gpt-oss:20b`.

### F1 — Harvest della base locale (1–2 giorni + run notturni)
- **Scopo:** base offline nei tuoi campi, con pilastri e frontiera.
- **Input:** `config.yaml` (topic, soglie).
- **Passi:**
  1. **Topic brevi** (1–3 termini, in OR tra sinonimi), es. `LoRaWAN`, `serverless edge`, `control barrier function`.
  2. **arXiv**: fino a 1000 candidati per topic, con pausa di 3 s tra le richieste.
  3. **Semantic Scholar batch** (anonimo): `citationCount`, `influentialCitationCount`, `year`, `venue`, `references`, `externalIds`.
  4. **IEEE OA**: Crossref (`prefix:10.1109`, ordinato per citazioni) + Unpaywall (`is_oa`).
  5. **Normalizzazione per età**: `cit_rate = citations / max(1, 2026 − year)`; percentile per topic.
  6. **Etichette:**
     - `pillar` se citations ≥ 100, oppure ≥ 50 per i topic di nicchia (soglia per topic in config);
     - `frontier` se year ≥ 2025 e percentile di `cit_rate` ≥ 80 nel suo topic.
  7. **Deduplica**: DOI, poi ID arXiv, poi titolo normalizzato (i preprint arXiv e la versione IEEE sono lo stesso paper).
  8. **Download PDF**: prima arXiv, poi `url_for_pdf` di Unpaywall; verificare l'header `%PDF`.
- **Output:** `data/db.sqlite` (papers, citations), `data/pdf/`.
- **Accettazione:**
  - almeno 500 pilastri e 500 di frontiera in totale;
  - ogni topic ha almeno 20 paper;
  - log con il conteggio per topic e per etichetta.

### F2 — Parsing PDF (1 giorno)
- **Scopo:** testo diviso per sezioni.
- **Passi:**
  1. PyMuPDF → testo; rimuovere intestazioni, piè di pagina e numeri di pagina.
  2. Riconoscere le sezioni con un'espressione regolare sui titoli (Abstract, Introduction, Related, Method, Experiments/Evaluation, Results, Discussion, Limitations, Conclusion, Future Work).
  3. Se il riconoscimento fallisce, usare l'LLM con la richiesta "segmenta in sezioni" sulle prime e ultime 3 pagine.
  4. Estrarre separatamente le **frasi chiave per i gap**: `limitation|future work|we leave|remains open|does not address|open problem`.
  5. Salvare `data/text/<id>.json` → `{sections: {...}, gap_sentences: [...], n_pages}`.
- **Accettazione:**
  - almeno il 90% dei PDF ha Introduction e Conclusion riconosciute;
  - almeno il 60% ha almeno una frase sui gap.

### F3 — Schede strutturate (2 giorni + run batch)
- **Scopo:** trasformare ogni paper in dati confrontabili.
- **Modello:** `gpt-oss:20b` (oppure `mistral-small:24b`), `num_ctx` 16k; input = abstract + intro + metodo + esperimenti + conclusione, troncati.
- **Schema `Card` (pydantic):**
  ```
  problem, motivation, contribution_type[method|system|benchmark|dataset|theory|survey|empirical-study],
  key_idea, assumptions[], method_family, baselines[], datasets[], metrics[],
  experimental_setup{hardware, scale, n_runs, stats_test}, main_result (con numeri),
  ablations[], limitations[], future_work[], threats_to_validity[],
  paper_structure[sezioni], title_pattern, claims[{claim, evidence_section}]
  ```
- **Passi:**
  1. Prompt in `extract/prompts/card.md` con 2 esempi few-shot scritti a mano.
  2. Batch con una coda e ripresa da dove si è interrotto (salta le schede già fatte).
  3. Controllo qualità: 30 schede verificate a mano, con una checklist di accuratezza campo per campo.
  4. Normalizzare i nomi di baseline, dataset e metriche (`dict` di alias + clustering degli embedding).
- **Output:** `data/cards/<id>.json` + tabella `cards`.
- **Accettazione:** accuratezza ≥ 85% sui campi `baselines`, `datasets`, `metrics`, `limitations` nel campione di 30.

### F4 — Indice (1 giorno)
- **Scopo:** recupero veloce per significato e per grafo delle citazioni.
- **Passi:**
  1. **Vettori:** `nomic-embed-text` su (a) titolo + abstract, (b) `key_idea`, (c) ogni frase sui gap. LanceDB con tre tabelle.
  2. **Grafo:** networkx o SQLite con archi `cites`; calcolare PageRank per topic, co-citazione e bibliographic coupling.
  3. **Mappa dei temi:** clustering (HDBSCAN) sugli embedding; per ogni cluster calcolare dimensione, età media, crescita (frontiera/pilastri) ed etichetta generata dall'LLM.
  4. **Tabelle di pratica per cluster:** baseline, dataset e metriche più usati; numero mediano di run; test statistici usati. È il "come si fa" del campo.
- **Output:** `data/vectors/`, `data/graph.gpickle`, `data/clusters.json`, `data/practice/<cluster>.json`.
- **Accettazione:** su 10 query scritte a mano, almeno 7 hanno un paper rilevante nei primi 5 risultati.

### F5 — Analisi dell'idea (½ giorno)
- **Input:** testo libero (anche 3 righe) + opzionale: venue target, vincoli di tempo.
- **Output `IdeaSpec`:**
  ```
  problem, domain, proposed_method, implicit_assumptions[], expected_contribution_type,
  keywords[], search_queries[5–10], risky_claims[]
  ```
- **Passo extra:** se l'idea è troppo vaga, genera **3 domande di chiarimento** (in chat) prima di continuare.
- **Accettazione:** le query generate recuperano almeno 3 paper pertinenti su 5 idee di test.

### F6 — Recupero su misura (2 giorni)
- **Scopo:** il corpus specifico per questa idea.
- **Passi:**
  1. **Locale:** ricerca vettoriale su (a) e (b) → top 50.
  2. **Live:** `search_queries` su arXiv + Semantic Scholar (`/paper/search`, anonimo) → nuovi candidati, che passano anche loro per F2–F3 al volo.
  3. **Espansione nel grafo:** per i 10 paper più vicini, un salto di riferimenti e citazioni; tenere quelli con co-citazione alta.
  4. **Riordinamento:** l'LLM valuta la pertinenza (0–3) su titolo + abstract; tenere quelli con punteggio ≥ 2.
  5. **Bilanciamento:** almeno il 30% pilastri e almeno il 30% frontiera.
  6. **Taglio temporale opzionale** (`--cutoff 2021-01-01`) per la valutazione F12.
- **Output:** `runs/<ts>/corpus.json` (50–200 paper con scheda).
- **Accettazione:** recall ≥ 0.6 sui "related work" reali dei paper usati come test.

### F7 — Novità e gap (2 giorni)
- **Novità:**
  1. I 5 paper più vicini all'`IdeaSpec`: confronto campo per campo (problema, metodo, assunzioni, setting).
  2. Verdetto: `già fatto` / `incrementale` / `differenza chiara` / `nuovo`, con le prove (ID dei paper + citazione testuale).
- **Mappa dei gap**, ogni gap con fonte e tipo:
  - **G1 dichiarati:** limitazioni e future work dei paper vicini, raggruppati e contati ("8 paper dicono X").
  - **G2 combinazioni mancanti:** coppie metodo × setting / metodo × dataset assenti nella matrice delle schede.
  - **G3 contraddizioni:** risultati discordanti sulla stessa metrica e lo stesso dataset.
  - **G4 assunzioni fragili:** assunzioni condivise da molti pilastri e messe in discussione dalla frontiera.
  - **G5 lacune di valutazione:** cluster senza un benchmark standard, con pochi run o senza test statistici.
  - **G6 trasferimento:** tecniche forti in un cluster vicino, non ancora applicate a questo.
- **Output:** `novelty.json`, `gaps.json`.
- **Accettazione:** ogni gap ha almeno 2 paper come prova; `cite_guard` passa al 100%.

### F8 — Direzioni di ricerca e punteggio (1–2 giorni)
- **Passi:**
  1. Generare 6–8 direzioni candidate, ognuna legata a uno o più gap.
  2. Per ognuna: domanda di ricerca, contributo atteso, perché ora, rischio principale.
  3. **Punteggio** (1–5, con motivazione e prove):

     | criterio | peso | fonte |
     |---|---|---|
     | Novità | 0.25 | distanza dai vicini (F7) |
     | Impatto potenziale | 0.20 | crescita del cluster, numero di paper che citano il gap |
     | Fattibilità | 0.25 | `resources.yaml` + costo degli esperimenti stimato dalle schede |
     | Rischio (inverso) | 0.15 | dipendenze, dati non disponibili, risultato negativo probabile |
     | Coerenza con il tuo profilo | 0.15 | similarità con i tuoi paper e progetti LD |
  4. Tenere le 3–5 migliori; i pesi sono modificabili in config.
- **Output:** `directions.json`.

### F9 — Revisore critico (1 giorno)
- **Scopo:** evitare idee deboli che sembrano buone.
- **Passi:**
  1. Un secondo prompt, nel ruolo di revisore severo di una rivista top, per ogni direzione: lavoro già esistente non visto (fa una nuova ricerca mirata), debolezze metodologiche, "so what?", esperimenti che un revisore chiederebbe.
  2. Se trova lavori precedenti non visti, li aggiunge al corpus e ripete F7 per quella direzione (massimo 1 iterazione).
  3. Il punteggio viene aggiornato; le direzioni con una fatal flaw vengono scartate, con la motivazione.
- **Output:** `critique.json`, ranking finale.

### F10 — Piano di ricerca completo (2–3 giorni)
- **Input:** la direzione scelta (la migliore, oppure quella indicata da te).
- **Sezioni del piano** (ognuna con citazioni dal corpus):
  1. **Domande di ricerca** (2–3) e **ipotesi falsificabili** (H1…Hn, ognuna con una metrica e una soglia).
  2. **Contributi attesi**, formulati come nei pilastri del cluster.
  3. **Posizionamento**: tabella di confronto con i 5–8 lavori più vicini.
  4. **Metodo**: architettura e algoritmo, scelte progettuali con alternative scartate.
  5. **Setup sperimentale**:
     - **Baseline:** quelle obbligatorie del cluster (dalle tabelle di pratica F4) + quelle più forti della frontiera.
     - **Dataset/benchmark:** quelli standard + uno che rende la tua valutazione diversa.
     - **Metriche:** primarie e secondarie, come nel cluster.
     - **Hardware:** mappato su `resources.yaml`.
  6. **Matrice degli esperimenti**: E1…En, ognuno legato a un'ipotesi; ablation; sensitivity; scalabilità.
  7. **Protocollo statistico**: numero di seed/run, intervalli di confidenza, test, correzione per confronti multipli.
  8. **Minacce alla validità** (interna, esterna, di costrutto) e come mitigarle.
  9. **Rischi e piano B**: per ogni rischio, un segnale precoce e un'alternativa.
  10. **Venue target**: 3 opzioni (dalle venue dei pilastri del cluster), con formato e limiti di pagine.
  11. **Scheletro del paper**: sezioni, figure e tabelle chiave, struttura del titolo e dell'abstract secondo i pattern dei pilastri.
  12. **Timeline**: milestone settimanali con criteri di uscita (go/no-go dopo gli esperimenti pilota).
  13. **Primo passo concreto**: cosa fare domattina (esperimento pilota da 1–2 giorni).
- **Output:** `plan.json` + `plan.md`.

### F11 — Report e chat (1–2 giorni)
- **Report:** `runs/<ts>/report.md` (e HTML opzionale) con sintesi, novità, mappa dei gap, direzioni con punteggio, critiche, piano completo, bibliografia (BibTeX dal corpus).
- **Chat:** `ra chat <run>`. Domande di approfondimento sul run ("perché scarti la D3?", "rifai il piano per IEEE T-ASE", "come cambia senza Coral?") con RAG sul corpus del run.
- **CLI:**
  - `ra idea "testo" [--venue X] [--cutoff DATA] [--weeks N]`
  - `ra ingest` (aggiorna la base)
  - `ra eval`

### F12 — Valutazione (2–3 giorni, poi continua)
- **E-A Test con data di taglio:**
  1. 20 idee ricavate dagli abstract di paper molto citati del 2022–2024, riformulate in modo vago.
  2. Si fa girare il sistema con `--cutoff` = 12 mesi prima della pubblicazione.
  3. Metrica: la direzione reale compare tra le prime 5? (hit@5); quante baseline e metriche reali il piano propone (recall).
- **E-B Test sui progetti LD:**
  1. Idea iniziale di LD1–LD11 (dai PLAN o dai primi commit).
  2. Confronto del piano con quanto fatto davvero e con le critiche dei revisori ricevute (quanti rilievi dei revisori il revisore critico di F9 aveva anticipato).
- **E-C Controllo citazioni:** percentuale di citazioni valide e pertinenti; obiettivo 100% valide e ≥ 90% pertinenti su un campione.
- **E-D Valutazione umana:** tu dai un voto da 1 a 5 su utilità, novità e fattibilità per 10 run.
- **Accettazione MVP:**
  - hit@5 ≥ 0.4;
  - recall delle baseline ≥ 0.6;
  - 0 citazioni inventate;
  - voto umano medio ≥ 3.5.

---

## 3. Roadmap

| Settimana | Fasi | Risultato |
|---|---|---|
| 1 | F0, F1 | base locale scaricata e etichettata (pilastri/frontiera) |
| 2 | F2, F3 | schede strutturate, controllo qualità sul campione di 30 |
| 3 | F4, F5, F6 | `ra idea` restituisce un corpus pertinente |
| 4 | F7, F8 | novità, gap e direzioni con punteggio |
| 5 | F9, F10, F11 | **MVP**: report completo end-to-end |
| 6 | F12 | valutazione, ritocco di prompt e pesi |
| poi | — | aggiornamento settimanale (timer systemd), chat, interfaccia web opzionale |

**Ordine alternativo "MVP veloce"** (risultato utile già alla settimana 2): F0 → F5 → F6 solo live (senza base locale) → F7 solo G1 → F8 → F10 → report. La base locale (F1–F4) si aggiunge dopo per qualità e velocità.

---

## 4. Rischi tecnici

| Rischio | Mitigazione |
|---|---|
| Rate limit di Semantic Scholar e arXiv senza key | batch, backoff, cache su disco, run notturni |
| PDF IEEE non scaricabili automaticamente | usare solo l'abstract + lo stesso paper su arXiv quando esiste |
| LLM locale debole nel ragionamento (F7–F10) | prompt in più passi con JSON stretto; `gpt-oss:20b` per il ragionamento; API opzionale dietro un flag |
| GPU condivisa occupata | coda con priorità bassa per F3; `num_ctx` controllato; riprendere dopo un'interruzione |
| Citazioni inventate | `cite_guard` + citazioni testuali obbligatorie per ogni affermazione |
| Corpus sbilanciato verso i temi più popolari | soglie per topic + bilanciamento in F6 |
| Il "successo" misurato in citazioni favorisce le mode | segnale di crescita (frontiera) + punteggio di coerenza con il profilo, non solo citazioni |

---

## 5. Decisioni aperte

1. LLM per il ragionamento (F7–F10): solo locale (`gpt-oss:20b`) oppure API opzionale.
2. Ordine: roadmap completa oppure "MVP veloce" prima.
3. Soglia per i topic di nicchia (proposta: 50 citazioni).
4. Lingua del report: italiano o inglese.

---

## 6. Stato implementazione (2026-10-18)

| Fase | Stato |
|---|---|
| F0 setup, CLI `./ra`, `resources.yaml` | fatto |
| F1 harvest (arXiv, Crossref+Unpaywall, S2 batch/riferimenti) | fatto; ricerca S2 anonima quasi sempre 429 → circuit breaker; arXiv limita se due processi lo interrogano insieme |
| F2 parsing, F3 schede (abstract/full) | fatto, provato su run reali |
| F4 base locale (`ra ingest --no-cards`): 12 topic, **1323 paper**, 30 cluster | fatto; schede della base non generate (la base serve per il recupero vettoriale) |
| F5–F11 `ra idea` end-to-end, ripresa del run, `ra chat` | fatto e provato (la chat ora conosce `resources.yaml`) |
| F12 valutazione | vedi sotto |

**Risultati F12**
- Recupero (le baseline reali sono nel corpus? match sul titolo, 5 casi con baseline): **0.60** (pool 0.60), dopo l'aggiunta della corsia "canonici" (paper molto citati in zona semantica) e dell'espansione dei riferimenti.
- Test con data di taglio, n=3 (prima della corsia canonici): direzione reale al rango 1 in 3/3.
- Test sui progetti LD (n=4: LD11, LD5, LD10, LD7): **hit@5 = 4/4** (ranghi 4, 1, 5, 2), citazioni valide 100%, citazioni testuali verificate 20/24.
- Run completo senza `--fast` (full1, 46 min): 9 schede da full text, `--venue` rispettato (prima opzione = venue richiesta), timeline di 12 settimane come richiesto, 0 citazioni non valide; i paper trovati dal revisore ora ricevono la scheda da abstract.
- Voto umano (E-D): **non ancora dato**: va fatto su idee vere.

**Limiti noti**
- La direzione 1 riformula l'idea, quindi hit@5 è più facile di quanto sembri; le idee dei casi sono già abbastanza specifiche.
- Il giudice è un LLM e il numero di casi è piccolo: i numeri sono indicativi.
- La conoscenza interna dell'LLM può contaminare il test con data di taglio (es. ha proposto "Transformer" come baseline prima del 2017).
- Un run completo dura 1–1,5 ore su GPU condivisa (`--fast` ~20–25 min).
