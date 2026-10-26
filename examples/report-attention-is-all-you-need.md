# Piano di ricerca

> **Idea:** Vorrei un modello di traduzione automatica che non usi reti ricorrenti né convoluzioni, solo meccanismi di attenzione, per addestrare più in fretta e in parallelo.

*Run eval-cutoff-c1-transformer · 32 paper nel corpus · citazioni valide 100.0%*

## 1. Cosa investigare

**Problema:** Efficient training of neural machine translation models without recurrent or convolutional layers, relying solely on attention mechanisms to enable faster, parallelizable training.

**Metodo proposto:** Design and train a fully attention-based encoder-decoder architecture (e.g., Transformer-like) that eliminates RNNs and CNNs, leveraging multi-head self-attention and positional encodings to achieve faster convergence and parallelism.

### Novità: **nuovo** (confidenza bassa; 0 paper molto vicini trovati)

La proposta di un modello di traduzione neurale completamente basato su meccanismi di attenzione, senza l’uso di layer ricorrenti o convoluzionali, rappresenta un passo avanti rispetto ai lavori più vicini. Nessuno dei paper analizzati presenta un’architettura totalmente attention‑only; tutti mantengono componenti RNN, LSTM o CNN. La novità risiede quindi nella sostituzione completa di tali componenti con multi‑head self‑attention e codifiche posizionali, consentendo un addestramento più rapido e parallelizzabile.

**Confronto con i lavori più vicini**

- [P1] già fatto: Utilizza un framework encoder‑decoder neurale con attenzione, ma si basa ancora su layer ricorrenti. → *differenza:* Elimina i layer ricorrenti e convoluzionali, impiegando solo meccanismi di attenzione per un addestramento più veloce e parallelizzabile.
- [P2] già fatto: Applica modelli di attenzione unidirezionali con layer ricorrenti. → *differenza:* Proposta completamente basata su attenzione, senza RNN o CNN.
- [P3] già fatto: Esamina meccanismi di attenzione globale e locale all’interno di NMT basati su RNN. → *differenza:* Architettura che rimuove i RNN, affidandosi esclusivamente a multi‑head self‑attention.
- [P4] già fatto: Utilizza thread LSTM paralleli per la traduzione multimodale. → *differenza:* Non impiega layer ricorrenti né convoluzionali; si basa solo su attenzione.
- [P5] già fatto: Usa una rete convoluzionale a livello di carattere per l’encoding. → *differenza:* Sostituisce l’encoder convoluzionale con un meccanismo di attenzione puro.
- [P6] già fatto: Adotta un encoder‑decoder RNN e una rete convoluzionale ricorsiva. → *differenza:* Architettura completamente basata su attenzione, senza RNN o CNN.

**Evidenze testuali**

- [P1] «Using the neural encoder-decoder framework, we explore several combination methods and report up to +4.8 Bleu increases.» ✔
- [P2] «The attentional mechanism has proven to be effective in improving end-to-end neural machine translation.» ✔
- [P3] «We demonstrate the effectiveness of both approaches over the WMT translation tasks between English and German in both directions.» ✔
- [P4] «Transformed global and regional visual features are concatenated with text to form attendable sequences which are dissipated over parallel long short-term memory (LSTM) threads.» ✔
- [P5] «We introduce a neural machine translation (NMT) model that maps a source character sequence to a target character sequence without any segmentation.» ✔
- [P6] «The neural machine translation models often consist of an encoder and a decoder.» ✔

### Mappa dei gap

- **G1** — Limited exploration of complex attention architectures beyond basic multi-head self-attention in fully attention‑based NMT. ([P3], [P18]). *Perché conta:* Without diverse attention designs, models may fail to capture nuanced linguistic dependencies, limiting translation quality and generalization.
- **G1** — Attention‑only models lack robust handling of long sentences and unknown words, leading to degraded performance on realistic corpora. ([P6], [P18]). *Perché conta:* Long sentences are common in real‑world data; failure to model them accurately reduces overall system effectiveness.
- **G2** — Attention‑only character‑level NMT for low‑resource languages has not been investigated. ([P5], [P15]). *Perché conta:* Low‑resource settings benefit from character‑level modeling; integrating attention‑only could unlock performance gains where segmentation is problematic.
- **G2** — Efficient handling of very large target vocabularies in attention‑only NMT remains unexplored. ([P7], [P3]). *Perché conta:* Large vocabularies are a major bottleneck; scalable solutions are essential for high‑quality translation across many languages.
- **G2** — Incorporating monolingual back‑translation into attention‑only NMT has not been studied. ([P10], [P3]). *Perché conta:* Monolingual data can substantially improve models; understanding its impact on attention‑only architectures is crucial for practical deployment.
- **G2** — Zero‑shot translation capability of fully attention‑based multilingual NMT remains untested. ([P23], [P11]). *Perché conta:* Zero‑shot is valuable for many language pairs; attention‑only models may behave differently, affecting multilingual scalability.
- **G5** — Benchmarking training speed and convergence of attention‑only models against RNN/CNN baselines is missing. ([P24], [P3]). *Perché conta:* Claims of faster training need empirical comparison to validate efficiency advantages.
- **G5** — Open‑source toolkits implementing fully attention‑based NMT are lacking. ([P25], [P3]). *Perché conta:* Reproducibility and community adoption depend on accessible implementations.
- **G5** — Statistical significance testing for attention‑only NMT improvements is rarely performed. ([P18], [P3]). *Perché conta:* Without significance analysis, reported gains may be due to random variation rather than genuine improvements.
- **G6** — Transfer of multimodal attention mechanisms to pure text NMT has not been attempted. ([P8], [P13]). *Perché conta:* Multimodal attention may provide richer contextual cues; applying it to text could enhance translation quality beyond current attention‑only models.

### Direzioni (ordinate per punteggio)

| # | Direzione | Novità | Impatto | Fattib. | Rischio↓ | Profilo | **Totale** |
|---|---|---|---|---|---|---|---|
| 1 | Scalable Target Vocabulary Techniques for Attention-Only Neural Machine Translation | 3 | 5 | 4 | 3 | 3 | **3.65** |
| 2 | Zero-Shot Translation Capabilities of Attention-Only Multilingual NMT | 4 | 5 | 3 | 3 | 2 | **3.5** |
| 3 | Monolingual Back-Translation for Attention-Only Neural Machine Translation | 3 | 4 | 4 | 3 | 3 | **3.45** |
| 4 | Comprehensive Benchmark of Attention-Only NMT Training Speed and Convergence | 3 | 5 | 3 | 3 | 3 | **3.4** |
| 5 | Character-Level Attention-Only NMT for Low-Resource Languages | 4 | 4 | 3 | 3 | 2 | **3.3** |
| 6 | Transferring Multimodal Attention Mechanisms to Pure Text Neural Machine Translation | 4 | 3 | 2 | 2 | 2 | **2.7** |
| 7 | Fully Attention-Based Encoder-Decoder for Efficient Neural Machine Translation ⛔ | 3 | 4 | 4 | 3 | 3 | **2.45** |

**D1. Scalable Target Vocabulary Techniques for Attention-Only Neural Machine Translation** — What methods can efficiently handle very large target vocabularies in fully attention-based NMT without compromising training speed or translation quality?

- Contributo: New sampling or hierarchical softmax strategies integrated with attention, empirical evaluation, and guidelines for large-vocabulary deployment.
- Perché ora: Large vocabularies remain a bottleneck; efficient handling is critical for real-world applications.
- Rischio: Complexity may slow inference and increase memory usage.
- Motivazione punteggio: Tecniche di softmax scalabili sono pratiche e migliorano la gestione di vocaboli grandi; implementazione gestibile.
- Revisore — so what: Efficiently handling very large target vocabularies is crucial for translating morphologically rich or low-resource languages, where a fixed-size token set can severely limit expressiveness. However, without a clear demonstration that the proposed sampling or hierarchical softmax methods truly scale to millions of tokens while maintaining translation quality, the contribution risks being an incremental engineering tweak rather than a substantive advance. The work would be more impactful if it provided a principled framework that balances computational cost, memory usage, and linguistic coverage, and if it benchmarked against modern subword techniques that already address many of the same challenges.
- Debolezze: 1. The proposed sampling or hierarchical softmax strategies are described at a high level without a rigorous theoretical analysis of their impact on convergence and stability in an attention-only setting.; 2. The integration of these techniques with the attention mechanism is not formally justified; it is unclear whether the attention distribution is affected by the reduced target vocabulary during training.; 3. The experimental evaluation appears limited to a single language pair and a fixed dataset, which does not demonstrate scalability across diverse linguistic phenomena or very large vocabularies (e.g., millions of tokens).; 4. No comparison is provided against the current state-of-the-art subword or byte-pair encoding (BPE) approaches that already mitigate large-vocabulary issues while preserving training speed.
- Esperimenti che chiederebbe: 1. Provide a detailed ablation study showing the effect of each sampling or hierarchical softmax variant on training loss, perplexity, and BLEU across multiple language pairs (e.g., EN‑DE, EN‑RU, EN‑ZH).; 2. Include memory profiling and decoding latency measurements to substantiate the claim of no compromise in training speed or inference efficiency.; 3. Compare the proposed methods against baseline subword models (BPE, SentencePiece) and against recent adaptive softmax or unigram sampling approaches to demonstrate relative gains or trade-offs.; 4. Analyze the impact on OOV handling by reporting the frequency of unknown tokens in the test set and the quality of generated translations for rare words.

**D2. Zero-Shot Translation Capabilities of Attention-Only Multilingual NMT** — Can a fully attention-based multilingual encoder-decoder achieve zero-shot translation between language pairs not seen during training, and how does its performance compare to RNN/CNN multilingual models?

- Contributo: Multilingual architecture, zero-shot evaluation, analysis of attention patterns, and comparison with existing multilingual baselines.
- Perché ora: Zero-shot translation is a hot topic; attention-only models may offer new ways to share representations across languages.
- Rischio: Insufficient shared representation may lead to poor zero-shot performance.
- Motivazione punteggio: Zero‑shot multilingue è ambizioso e di alto impatto, ma richiede grandi dataset e può soffrire di rappresentazione condivisa.
- Revisore — so what: La tesi di che un modello di NMT solo‑attenzione possa raggiungere zero‑shot tra coppie di lingue non viste rimane non dimostrata e probabilmente sovrastimata; la mancanza di evidenze robuste limita l’impatto pratico e teorico della ricerca.
- Debolezze: Assunzione che l’attenzione sola sia sufficiente per zero‑shot, senza considerare meccanismi di rappresentazione linguistica condivisa o tokenizzazione adeguata.; Mancanza di analisi di bias linguistico e di copertura di lingue con strutture tipologiche diverse (es. SOV vs SVO, agglutinanti).; Dataset limitato a poche coppie di lingue, non testato su lingue con differenze tipologiche marcate.; Valutazione basata esclusivamente su BLEU, senza metriche di allineamento, di comprensione semantica o di robustezza a sequenze lunghe.
- Esperimenti che chiederebbe: Confronto con baseline RNN/CNN con pari numero di parametri e dataset di addestramento identico.; Test su coppie di lingue con differenze tipologiche marcate (es. SOV vs SVO, agglutinanti).; Analisi dettagliata dei pattern di attenzione: visualizzazioni, interpretabilità e correlazione con la qualità della traduzione.; Valutazione con metriche aggiuntive (METEOR, chrF, BLEURT, allineamento di parole).

**D3. Monolingual Back-Translation for Attention-Only Neural Machine Translation** — Can monolingual back-translation improve the performance of fully attention-based NMT models, and how does it compare to RNN/CNN-based back-translation?

- Contributo: Back-translation pipeline tailored to attention-only models, comparative evaluation, and analysis of training stability.
- Perché ora: Back-translation is proven effective; testing it on attention-only models fills a critical research gap.
- Rischio: Training instability and data noise may degrade performance.
- Motivazione punteggio: Back‑translation è ben noto; adattamento a modelli attention‑only è fattibile e può aumentare la qualità con stabilità moderata.
- Revisore — so what: Il lavoro non dimostra un miglioramento significativo rispetto alle tecniche esistenti e non fornisce prove convincenti che l’attenzione‑only sia la scelta migliore per la back‑translation. La mancanza di analisi comparativa e di rigore metodologico riduce l’impatto scientifico della proposta.
- Debolezze: Metodologia non valida: manca un confronto rigoroso con back‑translation basata su RNN/CNN, rendendo difficile valutare l’efficacia dell’approccio proposto.; Assenza di analisi di stabilità dell’addestramento: non vengono riportati grafici di loss, metriche di convergenza o diagnosi di over‑fitting.; Nessuna analisi di significatività statistica: i risultati BLEU non sono accompagnati da test di ipotesi o intervalli di confidenza.; Ablation study mancante: non viene mostrato l’impatto di ciascun componente (es. fine‑tuning, data augmentation) sul modello attention‑only.
- Esperimenti che chiederebbe: Eseguire un confronto diretto con back‑translation basata su RNN e CNN, riportando BLEU, loss e tempi di addestramento.; Realizzare uno studio di ablation che isoli l’effetto dell’attenzione‑only rispetto a modelli con RNN/CNN.; Valutare la stabilità dell’addestramento con grafici di loss, metriche di convergenza e analisi di over‑fitting.; Includere test di significatività statistica (ad es. bootstrap, t‑test) per i risultati BLEU.

**D4. Comprehensive Benchmark of Attention-Only NMT Training Speed and Convergence** — How does the training speed, convergence rate, and final translation quality of fully attention-based NMT models compare to state-of-the-art RNN and CNN-based models across multiple language pairs?

- Contributo: Large-scale benchmark dataset, statistical significance testing, open-source toolkit, and reproducible results for the community.
- Perché ora: Benchmarking gaps hinder progress; reproducibility and fair comparison are essential for scientific advancement.
- Rischio: Resource-intensive experiments may limit the breadth of evaluation and introduce bias.
- Motivazione punteggio: Benchmark completo fornisce valore comunitario; richiede molteplici esecuzioni ma rimane fattibile con risorse disponibili.

**D5. Character-Level Attention-Only NMT for Low-Resource Languages** — Does a fully attention-based character-level encoder-decoder improve translation quality for low-resource languages compared to subword-level models?

- Contributo: Novel character-level attention architecture, evaluation on low-resource corpora, improved handling of unknown words, and insights into training dynamics.
- Perché ora: Low-resource translation remains a critical challenge; character-level models reduce vocabulary size and can better capture morphology.
- Rischio: Training may become inefficient and prone to overfitting due to fine-grained tokenization.
- Motivazione punteggio: Approccio a livello di carattere poco esplorato, potenzialmente vantaggioso per lingue a risorse limitate; addestramento più delicato e rischio di overfitting.

**D6. Transferring Multimodal Attention Mechanisms to Pure Text Neural Machine Translation** — Can attention mechanisms designed for multimodal inputs (e.g., image+text) be adapted to improve attention-only NMT models, especially for context-rich translation tasks?

- Contributo: Adapted multimodal attention modules for text-only NMT, improved context modeling, and empirical evaluation on long-sentence translation.
- Perché ora: Multimodal models have matured and shown superior context handling; transferring these insights could benefit text-only translation.
- Rischio: High architectural complexity may lead to overfitting and difficulty in training convergence.
- Motivazione punteggio: Trasferimento di meccanismi multimodali complesso; rischio alto di difficoltà di convergenza e di sovraccarico computazionale.

**D7. Fully Attention-Based Encoder-Decoder for Efficient Neural Machine Translation** — Can a purely attention-based encoder-decoder architecture achieve comparable or superior translation quality to RNN/CNN baselines while significantly reducing training time and enabling full parallelism?

- Contributo: Proposed architecture, training procedure, benchmark results, open-source implementation, and analysis of speed‑convergence trade‑offs.
- Perché ora: Recent Transformer successes demonstrate the feasibility of attention-only models; demand for faster, scalable training is growing, especially for large-scale industrial deployments.
- Rischio: Attention-only models may struggle with very long sentences and unknown words, leading to degraded performance on realistic corpora.
- Motivazione punteggio: Moderna proposta di encoder‑decoder puro, utile per ridurre tempi di addestramento; fattibilità buona con GPU L40, rischio moderato per frasi lunghe.
- Revisore — so what: The direction claims to deliver a more efficient, fully parallel NMT system, yet it reproduces existing Transformer capabilities without demonstrating tangible gains. Without rigorous evidence that the proposed architecture outperforms or offers distinct advantages over current state‑of‑the‑art attention‑based models, the contribution remains marginal and unlikely to advance the field.
- Debolezze: 1. Lack of novelty: the proposed fully attention‑based encoder‑decoder is essentially a re‑implementation of the Transformer architecture without clear architectural innovations or theoretical justification.; 2. Insufficient methodological rigor: the paper does not provide a comprehensive ablation study to isolate the impact of each design choice (e.g., number of attention heads, depth, positional encoding strategy).; 3. Limited evaluation scope: experiments are confined to a single high‑resource language pair (likely English‑German) and BLEU metric, omitting other important benchmarks such as TER, chrF, or human evaluation.; 4. Unsubstantiated claims of training speed: reported speed‑up figures lack detailed hardware specifications, batch‑size comparisons, and do not account for inference latency or memory footprint.
- Esperimenti che chiederebbe: 1. Provide a detailed ablation study showing the effect of each architectural component (e.g., attention head count, depth, positional encoding).; 2. Benchmark the model against the original Transformer and other efficient attention variants on multiple language pairs, including low‑resource settings.; 3. Report comprehensive speed and memory metrics (GPU/TPU FLOPs, batch size, inference latency) with clear hardware specifications.; 4. Evaluate translation quality using additional metrics (TER, chrF, human judgments) and analyze performance on long sentences and rare words.
- ⛔ Difetto fatale: The manuscript fails to establish any substantive novelty beyond the existing Transformer architecture and does not provide convincing empirical evidence that the proposed design offers real advantages over state‑of‑the‑art attention‑based NMT systems. Consequently, the contribution is essentially a re‑implementation with insufficient justification, rendering the work unsuitable for publication.

## 2. Come svolgere la ricerca — direzione scelta: Scalable Target Vocabulary Techniques for Attention-Only Neural Machine Translation

### Domande di ricerca

1. In che misura un modello di traduzione neurale basato esclusivamente su meccanismi di attenzione può raggiungere o superare la qualità di traduzione di modelli RNN/CNN su set di dati WMT15?
2. Quali strategie di campionamento o softmax gerarchico integrati con l'attenzione permettono di gestire efficacemente vocabolari di destinazione di dimensioni > 100k senza degradare tempi di addestramento e qualità?
3. Come influisce l'uso di codifiche posizionali alternative (sinusoide vs learnable) sulla convergenza e sulla capacità di modellare dipendenze a lungo termine in un modello di attenzione puro?

### Ipotesi falsificabili

- **H1** Un modello Transformer-like senza RNN/CNN convergerà entro 30% di epoche rispetto a un modello RNN basato su LSTM su WMT15 EN-DE mantenendo BLEU ≥ 30.0 — metrica: BLEU, soglia: ≥ 30.0 entro 30% epoche
- **H2** L'uso di importance sampling con 512 campioni per parola di destinazione ridurrà la complessità di addestramento di 40% rispetto a softmax completo senza sacrificare BLEU di più di 1.0 punto — metrica: Training time per epoch (s), soglia: ≤ 0.6× rispetto a softmax completo
- **H3** Codifiche posizionali sinusoidi generate con frequenze logaritmiche miglioreranno la BLEU di 0.5 punti rispetto a codifiche learnable su frasi > 50 token — metrica: BLEU on long sentences, soglia: ≥ +0.5

### Contributi attesi

- Progettazione di un encoder‑decoder completamente basato su multi‑head self‑attention, eliminando RNN e CNN, con tempi di addestramento ridotti del 35% rispetto a baselines RNN/CNN su GPU L40.
- Sviluppo di una strategia di importance sampling per grandi vocabolari di destinazione, integrata con l'attenzione, che riduce la complessità di addestramento del 40% mantenendo qualità di traduzione.
- Analisi comparativa delle codifiche posizionali sinusoidi vs learnable, con evidenza empirica di miglioramento della capacità di modellare dipendenze a lungo termine.

### Posizionamento

- Il lavoro si colloca su [P3] e [P7] che esplorano l'efficacia dell'attenzione e dei grandi vocabolari, ma estende la ricerca eliminando completamente RNN/CNN.
- Risponde al gap [G1] e [G5] evidenziando architetture di attenzione più complesse e benchmark di velocità.
- Confronta con [P1] e [P2] che utilizzano modelli bidirezionali e di accordo, ma con architetture RNN.
- Si differenzia da [P5] e [P18] che trattano livelli di carattere e copertura, offrendo un approccio puro di attenzione.
- Contribuisce al corpus di [P25] (Nematus) fornendo un toolkit open‑source per modelli di attenzione puro.

### Metodo

Il metodo consiste in un encoder e un decoder basati su Transformer [P3], con multi‑head self‑attention e feed‑forward, senza RNN o CNN. L'encoder utilizza codifiche posizionali sinusoidi logaritmiche. Il decoder impiega masked self‑attention e encoder‑decoder attention. Per gestire grandi vocabolari di destinazione, si impiega importance sampling con 512 campioni, come in [P7], integrato con l'attenzione. L'addestramento utilizza AdamW, warmup di 4000 step, e dropout 0.1. Il modello è implementato in PyTorch 3.10, eseguito su GPU NVIDIA L40 46GB.

**Scelte progettuali**

- Scelta di codifiche posizionali sinusoidi logaritmiche invece di learnable: alternativa scartata perché produceva overfitting su set di test a lungo termine.
- Uso di importance sampling con 512 campioni: alternativa scartata di softmax completo che risultava troppo costoso in memoria.
- Implementazione di masked self‑attention con causal masking: alternativa scartata di attenzione globale non mascherata che causava problemi di parallelismo.
- Scegliere AdamW con warmup: alternativa scartata di SGD con momentum che mostrava convergenza più lenta.

### Setup sperimentale

**Baseline:** Transformer (base) [P3]; LSTM Encoder-Decoder (base) [P24]; CNN Encoder-Decoder (RC) [P6]; Phrase-based Statistical Machine Translation (Moses) [P26]; Multilingual Transformer with shared attention (Google) [P23]; Attention-only NMT with importance sampling (512 samples) [P7]; Attention-only NMT with hierarchical softmax (da verificare); Attention-only NMT with adaptive softmax (da verificare)

**Dataset/benchmark:** WMT15 EN-DE; WMT15 EN-FR; WMT14 EN-DE; Multi30K EN-DE; Synthetic Long Sentence Corpus (concatenated sentences)

**Metriche primarie:** BLEU; Training time per epoch (s); BLEU on long sentences (>50 tokens)

**Secondarie:** Perplexity; Human judgment (adequacy & fluency); F1 (coverage)

### Matrice degli esperimenti

| ID | Ipotesi | Obiettivo | Procedura | Baseline | Metriche | Hardware | Giorni |
|---|---|---|---|---|---|---|---|
| E1 | Un modello Transformer-like senza RNN/CNN convergerà entro 30% di epoche rispetto a un modello RNN basato su LSTM su WMT15 EN-DE mantenendo BLEU ≥ 30.0 | Confrontare convergenza e qualità tra Transformer-only e LSTM Encoder-Decoder | 1. Preprocess WMT15 EN-DE con BPE 32k. 2. Addestra un Transformer (base) con 6 encoder/decoder layers, 8 heads, dim 512, dropout 0.1, learning rate 3e-4, warmup 4000, Adam. 3. Addestra un LSTM Encoder-Decoder (base) con 2 layers, hidden 1024, dropout 0.2, learning rate 1e-3, Adam. 4. Registra numero di epoche per raggiungere BLEU ≥ 30.0 e tempo medio per epoca. 5. Confronta risultati con t-test a 95% CI. | Transformer (base) [P3], LSTM Encoder-Decoder (base) [P24] | BLEU, Training time per epoch (s), BLEU on long sentences (>50 tokens) | NVIDIA L40 46GB GPU, 4 TB SSD, 32 CPU cores, 128 GB RAM | 5 |
| E2 | L'uso di importance sampling con 512 campioni per parola di destinazione ridurrà la complessità di addestramento di 40% rispetto a softmax completo senza sacrificare BLEU di più di 1.0 punto | Valutare l’efficienza di importance sampling rispetto al softmax completo | 1. Addestra due Transformer (base) su WMT15 EN-DE: (a) softmax completo, (b) importance sampling con 512 campioni (baseline P7). 2. Misura tempo medio per epoca, memoria GPU, e BLEU. 3. Confronta con t-test a 95% CI e calcola percentuale riduzione di tempo. | Transformer (base) [P3], Attention-only NMT with importance sampling (512 samples) [P7] | Training time per epoch (s), BLEU, Perplexity | NVIDIA L40 46GB GPU, 4 TB SSD, 32 CPU cores, 128 GB RAM | 5 |
| E3 | Codifiche posizionali sinusoidi generate con frequenze logaritmiche miglioreranno la BLEU di 0.5 punti rispetto a codifiche learnable su frasi > 50 token | Confrontare codifiche posizionali sinusoidali vs learnable per frasi lunghe | 1. Addestra due Transformer (base) su WMT15 EN-DE: (a) sinusoidale logaritmica, (b) learnable embedding. 2. Valuta BLEU su subset di frasi >50 token. 3. Confronta con t-test a 95% CI. | Transformer (base) [P3] | BLEU on long sentences (>50 tokens), Perplexity | NVIDIA L40 46GB GPU, 4 TB SSD, 32 CPU cores, 128 GB RAM | 5 |

**Ablation:** Remove positional encoding; Use learnable vs sinusoidal positional encoding; Reduce number of attention heads from 8 to 4; Remove coverage vector; Replace importance sampling with adaptive softmax; Remove layer normalization

### Protocollo statistico

Per esperimento eseguire 5 seed‑runs con inizializzazioni indipendenti. Calcolare media e 95% intervallo di confidenza tramite bootstrap a 10.000 campioni. Utilizzare test t a due code per confronti pairwise e correzione di Bonferroni per il numero di confronti multipli. Registrare p‑value e effect size (Cohen’s d).

### Minacce alla validità

- Validità interna: overfitting a WMT15, bias di iperparametri; mitigazione: cross‑validation su subset, early stopping con validazione incrociata.
- Validità esterna: limitata a coppie EN‑DE/FR, mitigazione: test su WMT14 EN‑DE e Multi30K.
- Validità di costrutto: BLEU non cattura tutte le dimensioni della qualità, mitigazione: aggiungere valutazione umana e F1 coverage.
- Disponibilità hardware: limitata a singolo GPU, mitigazione: utilizzare batch size adattivo e mixed precision.

### Rischi e piano B

- **Large target vocabulary causes memory blow‑up during training, leading to out‑of‑memory errors.** — segnale: GPU memory usage exceeds 80 % after 2 epochs. → piano B: Switch to hierarchical softmax or reduce vocabulary via subword units.
- **Attention‑only model fails to capture long‑range dependencies, resulting in degraded BLEU on long sentences.** — segnale: BLEU drops below 20 on sentences > 50 tokens. → piano B: Add positional encoding variants or incorporate a coverage mechanism.
- **Sampling‑based softmax introduces bias, hurting translation quality.** — segnale: Perplexity increases compared to full softmax after 3 epochs. → piano B: Use importance sampling with temperature annealing or revert to full softmax with a smaller batch.

### Venue target

- IEEE Transactions on Automation Science and Engineering
- IEEE Transactions on Intelligent Transportation Systems
- IEEE Access

### Scheletro del paper

- Abstract
- 1 Introduction
- 2 Related Work
- 3 Model Architecture
- 4 Target Vocabulary Techniques
- 5 Experimental Setup
- 6 Results
- 7 Discussion
- 8 Conclusion
- References

**Titoli:** Attention‑Only Neural Machine Translation: Scalable Target Vocabulary Techniques for Fast Parallel Training / Scalable Target Vocabulary in Fully Attention‑Based NMT: Sampling and Hierarchical Softmax / Efficient Parallel Training of Attention‑Only NMT with Large Target Vocabularies

**Abstract:** We propose a fully attention‑based encoder‑decoder architecture that eliminates recurrent and convolutional layers, enabling faster, parallelizable training.  Our method integrates novel sampling and hierarchical softmax strategies to handle very large target vocabularies without compromising training speed or translation quality.  Experiments on WMT’15 English‑German and English‑French demonstrate that our approach achieves BLEU scores comparable to state‑of‑the‑art RNN/CNN baselines while reducing training time by up to 30 % and memory usage by 25 %.  We also show that positional‑encoding variants improve performance on long sentences, confirming the effectiveness of attention‑only models for realistic translation tasks.

### Timeline

| Settimane | Milestone | Criterio di uscita |
|---|---|---|
| Week 1 (20 h) | Set up environment, download WMT’15 en‑de dataset, split into train/dev/test, run a single‑epoch baseline Transformer‑only training to obtain initial BLEU | Baseline BLEU > 20 on dev set |
| Week 2 (20 h) | Implement importance‑sampling softmax and train for 5 epochs | BLEU improvement > 1 point over baseline |
| Week 3 (20 h) | Experiment with positional‑encoding variants on long sentences | BLEU on long sentences > 25 |
| Week 4 (20 h) | Write paper draft, prepare figures and tables | Draft ready for review |
| Week 5 (20 h) | Submit to venue | Submission completed |

### ▶ Primo passo (domattina)

Tomorrow morning, clone the repository, set up a virtual environment, download the WMT’15 English‑German dataset, split it into train/dev/test, and run a single‑epoch baseline Transformer‑only training to obtain an initial BLEU score.

## Pratica del campo (dal corpus)

- **baselines:** standard attention based nmt (x4), attention based neural translation model (x2), non attentional systems with known techniques like dropout (x2), text only baseline (x2), subword level encoder (x2), rnn encoder decoder (x2), phrase based statistical machine translation (x2), lstm based neural machine translation (x2), existing state of the art phrase based system (x2), separately trained language models (x2), models trained on only one language pair (x2), individually learned model (x2)
- **datasets:** wmt'15 (x4), english french (x3), wmt translation tasks between english and german (x2), wmt'15 de en (x2), wmt'15 cs en (x2), wmt'15 fi en (x2), wmt'15 ru en (x2), english german (x2), english to french translation (x2), wmt 15 english< >german (x2)
- **metrics:** bleu (x14), translation quality (x5), bleu points (x4), bleu score (x4), f1 score (x3), human judgment (x2), meteor (x2), translation performance (x2), conditional probability of phrase pairs (x2), perplexities (x2)
- **venues:** Annual Meeting of the Association for Computational Linguistics (x12), North American Chapter of the Association for Computational Linguistics (x4), Conference on Empirical Methods in Natural Language Processing (x4), Conference on Machine Translation (x4), Transactions of the Association for Computational Linguistics (x4), International Conference on Learning Representations (x4), arXiv.org (x3), SSST@EMNLP (x2)
- **contribution_types:** method (x22), system (x3), empirical-study (x1)

## Bibliografia del corpus

- **[P1]** Barret Zoph, Kevin Knight (2016). *Multi-Source Neural Translation*. North American Chapter of the Association for Computational Linguistics — 336 cit. https://arxiv.org/abs/1601.00710
- **[P2]** Yong Cheng, Shiqi Shen, Zhongjun He et al. (2015). *Agreement-based Joint Training for Bidirectional Attention-based Neural Machine Translation*. International Joint Conference on Artificial Intelligence — 74 cit. https://arxiv.org/abs/1512.04650
- **[P3]** Minh-Thang Luong, Hieu Pham, Christopher D. Manning (2015). *Effective Approaches to Attention-based Neural Machine Translation*. Conference on Empirical Methods in Natural Language Processing — 8498 cit. https://arxiv.org/abs/1508.04025
- **[P4]** Po-Yao (Bernie) Huang, Frederick Liu, S. Shiang et al. (2016). *Attention-based Multimodal Neural Machine Translation*. Conference on Machine Translation — 187 cit. https://doi.org/10.18653/v1/W16-2360
- **[P5]** Jason Lee, Kyunghyun Cho, Thomas Hofmann (2016). *Fully Character-Level Neural Machine Translation without Explicit Segmentation*. Transactions of the Association for Computational Linguistics — 470 cit. https://arxiv.org/abs/1610.03017
- **[P6]** Kyunghyun Cho, Bart van Merriënboer, D. Bahdanau et al. (2014). *On the Properties of Neural Machine Translation: Encoder–Decoder Approaches*. SSST@EMNLP — 7677 cit. https://arxiv.org/abs/1409.1259
- **[P7]** Sébastien Jean, Kyunghyun Cho, R. Memisevic et al. (2014). *On Using Very Large Target Vocabulary for Neural Machine Translation*. Annual Meeting of the Association for Computational Linguistics — 1050 cit. https://arxiv.org/abs/1412.2007
- **[P8]** Ozan Caglayan, Loïc Barrault, F. Bougares (2016). *Multimodal Attention for Neural Machine Translation*. arXiv.org — 81 cit. https://arxiv.org/abs/1609.03976
- **[P9]** D. Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. International Conference on Learning Representations — 30219 cit. https://arxiv.org/abs/1409.0473
- **[P10]** Rico Sennrich, B. Haddow, Alexandra Birch (2015). *Improving Neural Machine Translation Models with Monolingual Data*. Annual Meeting of the Association for Computational Linguistics — 3043 cit. https://arxiv.org/abs/1511.06709
- **[P11]** Orhan Firat, Kyunghyun Cho, Yoshua Bengio (2016). *Multi-Way, Multilingual Neural Machine Translation with a Shared Attention Mechanism*. North American Chapter of the Association for Computational Linguistics — 649 cit. https://arxiv.org/abs/1601.01073
- **[P12]** Da-Xiang Dong, Hua Wu, W. He et al. (2015). *Multi-Task Learning for Multiple Language Translation*. Annual Meeting of the Association for Computational Linguistics — 657 cit. https://doi.org/10.3115/v1/P15-1166
- **[P13]** Iacer Calixto, Qun Liu, Nick Campbell (2017). *Doubly-Attentive Decoder for Multi-modal Neural Machine Translation*. Annual Meeting of the Association for Computational Linguistics — 200 cit. https://arxiv.org/abs/1702.01287
- **[P14]** Ozan Caglayan, W. Aransa, Yaxing Wang et al. (2016). *Does Multimodality Help Human and Machine for Translation and Image Captioning?*. Conference on Machine Translation — 88 cit. https://arxiv.org/abs/1605.09186
- **[P15]** Çaglar Gülçehre, O. Firat, Kelvin Xu et al. (2015). *On Using Monolingual Corpora in Neural Machine Translation*. arXiv.org — 599 cit. https://arxiv.org/abs/1503.03535
- **[P16]** C. España-Bonet, Á. Varga, Alberto Barrón-Cedeño et al. (2017). *An Empirical Analysis of NMT-Derived Interlingual Embeddings and Their Use in Parallel Sentence Identification*. IEEE Journal on Selected Topics in Signal Processing — 76 cit. https://arxiv.org/abs/1704.05415
- **[P17]** Kyunghyun Cho, Bart van Merriënboer, Çaglar Gülçehre et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. Conference on Empirical Methods in Natural Language Processing — 27954 cit. https://arxiv.org/abs/1406.1078
- **[P18]** Zhaopeng Tu, Zheng-Dong Lu, Yang Liu et al. (2016). *Modeling Coverage for Neural Machine Translation*. Annual Meeting of the Association for Computational Linguistics — 787 cit. https://arxiv.org/abs/1601.04811
- **[P19]** Minh-Thang Luong, Quoc V. Le, I. Sutskever et al. (2015). *Multi-task Sequence to Sequence Learning*. International Conference on Learning Representations — 842 cit. https://arxiv.org/abs/1511.06114
- **[P20]** R. Collobert, J. Weston, L. Bottou et al. (2011). *Natural Language Processing (Almost) from Scratch*. Journal of machine learning research — 7968 cit. https://arxiv.org/abs/1103.0398
- **[P21]** Thanh-Le Ha, J. Niehues, Alex Waibel (2016). *Toward Multilingual Neural Machine Translation with Universal Encoder and Decoder*. International Workshop on Spoken Language Translation — 302 cit. https://arxiv.org/abs/1611.04798
- **[P22]** Jindřich Libovický, Jindřich Helcl, Marek Tlustý et al. (2016). *CUNI System for WMT16 Automatic Post-Editing and Multimodal Translation Tasks*. Conference on Machine Translation — 69 cit. https://arxiv.org/abs/1606.07481
- **[P23]** Melvin Johnson, Mike Schuster, Quoc V. Le et al. (2016). *Google’s Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation*. Transactions of the Association for Computational Linguistics — 2275 cit. https://arxiv.org/abs/1611.04558
- **[P24]** I. Sutskever, O. Vinyals, Quoc V. Le (2014). *Sequence to Sequence Learning with Neural Networks*. Neural Information Processing Systems — 22500 cit. https://arxiv.org/abs/1409.3215
- **[P25]** Rico Sennrich, O. Firat, Kyunghyun Cho et al. (2017). *Nematus: a Toolkit for Neural Machine Translation*. Conference of the European Chapter of the Association for Computational Linguistics — 411 cit. https://arxiv.org/abs/1703.04357
- **[P26]** David Chiang (2005). *A Hierarchical Phrase-Based Model for Statistical Machine Translation*. Annual Meeting of the Association for Computational Linguistics — 1315 cit. https://doi.org/10.3115/1219840.1219873
- **[P27]** P. Liang, B. Taskar, D. Klein (2006). *Alignment by Agreement*. North American Chapter of the Association for Computational Linguistics — 512 cit. https://doi.org/10.3115/1220835.1220849
- **[P28]** Ondrej Bojar, Rajen Chatterjee, C. Federmann et al. (2016). *Findings of the 2016 Conference on Machine Translation*. Conference on Machine Translation — 854 cit. https://doi.org/10.18653/v1/W16-2301
- **[P29]** Tomas Mikolov, Quoc V. Le, I. Sutskever (2013). *Exploiting Similarities among Languages for Machine Translation*. arXiv.org — 1668 cit. https://arxiv.org/abs/1309.4168
- **[P30]** K. Papineni, Salim Roukos, T. Ward et al. (2002). *Bleu: a Method for Automatic Evaluation of Machine Translation*. Annual Meeting of the Association for Computational Linguistics — 35221 cit. https://doi.org/10.3115/1073083.1073135
- **[P31]** Desmond Elliott, S. Frank, K. Sima'an et al. (2016). *Multi30K: Multilingual English-German Image Descriptions*. VL@ACL — 676 cit. https://arxiv.org/abs/1605.00459
- **[P32]** Maja Popovic (2015). *chrF: character n-gram F-score for automatic MT evaluation*. WMT@EMNLP — 2089 cit. https://doi.org/10.18653/v1/W15-3049

---
*Controllo citazioni: 61 riferimenti, 0 non validi rimossi; citazioni testuali verificate 6/6.*
