# eit-clinical — Learning Journal

Quaderno di studio personale di Riccardo. Aggiornato ad ogni sessione di lavoro.

---

## Sessione 2026-03-27 — GREIT bridge, spatial reconstruction diagnosis, ML strategy, docs split

### Tasks worked on
- Fase 2 wrap-up: `eit_greit.py` (GREIT bridge), notebook 03 validation, docs split
- Branch `feat/eit-parser-complete` brought to final state before PR

### Concepts explained

- **GREIT is a linear reconstruction**: `pixel = R_matrix @ vv` where R (1024×208) is
  built from a FEM sensitivity model. Data-driven approach = estimating R empirically
  from paired `.eit`/`.bin` data instead of using the FEM model.
- **`std` vs `fmmu` protocol ordering**: pyEIT generates 208 pairs in two orderings.
  `std` always starts the measurement sweep from electrode 0 (absolute). `fmmu` starts
  from the active drive electrode A (rotated). Both give 208 pairs; they differ from
  drive pattern 2 onward. Dräger hardware likely uses `fmmu` (natural sweep from the
  electrode adjacent to the sink). Wrong protocol → spatially scrambled image even though
  the global signal sum is correct (sum is order-independent).
- **Why global signal is correct but image is wrong**: `nansum(images, axis=(1,2))` sums
  all pixels regardless of spatial position. If measurements are assigned to wrong electrode
  pairs, each pixel is wrong but the total stays the same. This is why temporal phase was
  correct while bilateral lung structure was absent.
- **eitprocessing does NOT read `.eit`**: only reads `.bin` (already-reconstructed 32×32
  from the device firmware). FastEIT is the only open-source Python parser for `.eit`.
  This is the competitive differentiator.
- **ML reconstruction proposal**: use paired `.eit`/`.bin` frames as (X, y) to learn
  the Dräger reconstruction matrix empirically. Simplest model: Ridge regression
  (`sklearn`), which estimates R (1024×208) from data — equivalent to GREIT but without
  the FEM model. Training on 11k frames takes seconds. Neural network (CNN/U-Net) is
  optional and probably not needed if the transform is linear.
- **Frame alignment prerequisite**: before building the ML dataset, verify that frame k
  in `.eit` corresponds to frame k in `.bin` (lag=0 cross-correlation). This is checked
  in notebook §10.
- **EELI and device recalibration**: absolute impedance baseline shifts after each
  PulmoVista recalibration. Cross-recording comparison requires a shared external
  `ref_frame` (mean of a stable baseline period at the start of the session). Already
  implemented in `reconstruct_greit(ref_frame=np.ndarray)`.

### Riccardo's questions and insights
- "Perché non ci vogliono mesi per il ML?" → Training time is hours at most (Ridge =
  seconds, CNN = hours on CPU). Months was a mistake — I was thinking about full feature
  development, not training time.
- "Possiamo fare un modello che impara quello che fa Dräger?" → Yes, this is called
  data-driven reconstruction. Ridge regression is the right starting point (linear EIT).
- "Questo mi porterebbe più nel mio campo" → Exactly right: the ML angle is clinically
  original (no one else has paired `.eit`/`.bin` data from open-source parsing) and
  positions the work as applied research, not physics engineering.

### Epiphanies
- The `.bin` is NOT raw data — it is already reconstructed by the device. eitprocessing
  works at pixel level, never seeing the 208 measurements. FastEIT is unique precisely
  because it reads both layers.
- `std` vs `fmmu` is empirically testable in the notebook: if `fmmu` shows bilateral
  lung structure and `std` does not, the ordering is confirmed.

### Technical notes
- `eit_greit.py`: sign negation + rot90(k=1) applied inside `reconstruct_greit()` so
  callers always get impedance convention + anatomical orientation without manual steps.
- External `ref_frame` is the correct mechanism for cross-recording EELI comparison.
  Documented in the docstring with a usage example.
- docs split: `data_model.md` = containers only; `parsers.md` = parser reference
  (new file, comprehensive); `parsing_layer.md` = architecture + extension recipes.

### Branch status
- `feat/eit-parser-complete`: commits 1-3 done (eit_greit, notebook, docs)
- Next: §9 std/fmmu test and §10 cross-correlation to be run by Riccardo in the notebook
- After PR merge to dev: open `feat/ml-reconstruction` from dev

---

## Sessione 2026-03-24 — Git workflow, CI/CD, benchmark notebook

### Task lavorate
- B.1: Notebook benchmark `np.memmap` vs `struct.unpack_from`
- D.3 (completamento): README badge table verticale
- CI.1 (completamento): build workflow + test/lint trigger su dev
- CI.2 parziale: codecov setup (pytest-cov, codecov-action@v5, CODECOV_TOKEN)
- P.1: GitHub profile README

### Concetti spiegati

- **Git branching model (trunk-based semplificato)**: `main` = released/stable, `dev` = working environment, feature branch da dev per ogni task. Il branch default su GitHub punta a quello che vede il pubblico.
- **`git remote prune origin`**: rimuove i riferimenti locali a branch remoti già cancellati su GitHub (dopo merge e delete PR).
- **`np.memmap` e RAM peak**: il mapping OS vive fuori dall'heap Python — `tracemalloc` non vede i byte grezzi del file. Il RAM peak misurato è l'output array `(N, 32, 32)` float32 creato da `replace_no_data_sentinels` + la maschera booleana intermedia. I campi non acceduti (ts, Medibus) non entrano mai nell'heap.
- **Speedup ratio che decresce a file grandi**: a file piccoli la CI/CD è bottlenecked da Python overhead (loop, tuple, function calls). A file grandi entrambi gli approcci diventano I/O-bound — il disco diventa il collo di bottiglia comune, e il vantaggio relativo si riduce.
- **GitHub Actions badge URL e branch**: `badge.svg` senza parametri usa il default branch. `branch/main` punta a main, `branch/dev` a dev. Se il coverage è uploadato da dev ma il badge punta a main, mostra `unknown`.
- **Codecov**: riceve il `coverage.xml` generato da `pytest --cov --cov-report=xml` tramite `codecov-action`. Da ottobre 2023 richiede token anche per repo pubbliche. Il token va come GitHub Secret, mai hardcoded nel yaml.
- **Standard README badge table**: layout verticale (label | badge) è lo standard della community Python open source (pandas, sklearn, requests lo fanno tutti). Non specifico di eitprocessing.
- **`fail_ci_if_error: false`**: il job CI non fallisce se l'upload a codecov non riesce — evita che un problema esterno rompa la CI.

### Domande di Riccardo
- "perche stiamo caricando in ram con mmap comunque?" → Il RAM che misuri è l'output array (pixels puliti), non i byte del file — quello è gestito dall'OS fuori dall'heap Python.
- "la struttura mia e quella di eitprocessing e lo stesso standard?" → Sì, separation of concerns (models/parsers), entry point unico, dtype numpy per binari, BaseParser — tutti pattern standard, non copiati.
- "coverage unknown perché?" → Badge puntava a `branch/main` ma il coverage era stato uploadato da `dev`. Fix: cambiare branch nel badge URL.

### Epifanie e insight
- "il badge unknown non è un errore di codecov, è solo che il branch nel link è sbagliato"
- "pytest-cov era già in pyproject.toml da prima — avevamo già tutto pronto senza saperlo"

### Note tecniche
- `codecov/codecov-action@v5` (non v4) è la versione attuale raccomandata da codecov
- `if: matrix.python-version == '3.12'` evita 3 upload duplicati dalla matrix 3×Python
- `--cov-branch` abilita il branch coverage (copertura dei rami if/else), non solo line coverage

---

## Sessione 2026-03-27 — Parser .eit completo + visione scientifica multicentrica

### Task lavorate
- Completamento `DragerEitParser.parse()`: memmap frame data, estrazione tutti i campi rilevanti
- Discussione architettura `RawImpedanceData`: split `aux_signals` → `signals` + `clinical`
- Discussione bridge pyEIT e standardizzazione cross-vendor

### Concetti spiegati

- **`RawImpedanceData` split semantico**: i campi del frame si dividono in due categorie con significato diverso.
  - `signals`: segnali fisici EIT per-frame (`timestamp`, `trans_A`, `trans_B`, `injection_current`, `voltage_A`, `voltage_B`) — quello che la macchina misura elettricamente
  - `clinical`: dati di contesto clinico per-frame (`medibus` 67 canali ventilatore, `event_text`) — quello che il paziente/ventilatore sta facendo
  - `measurements`: il campo core per pyEIT = `vv` calibrato (Task 2.4.2)

- **pyEIT non "sa" quanti elettrodi hai**: il protocollo (`n_el`, `dist_exc`, `step_meas`) si configura esplicitamente. La matrice `vv` deve avere tante colonne quante misure definisce quel protocollo. Dräger 16 el → 208 misure, Timpel 32 el → misure diverse.

- **GREIT (Graz Consensus Reconstruction Algorithm)**: algoritmo standardizzato per ricostruzione EIT clinica 2D. Parametri raccomandati da consensus paper (Adler 2009). Permette di giustificare scientificamente le scelte algoritmiche senza dover derivare tutto from scratch.

- **Geometria EIT sempre circolare**: sia Dräger che Timpel modellano il torace come sezione circolare 2D. È una semplificazione nota di tutti i sistemi EIT — funziona perché la cintura va sempre al 4° spazio intercostale. Non è una differenza tra vendor.

- **Pixel vs ROI: cosa è confrontabile cross-vendor**:
  - Pixel singoli: NON confrontabili (16 el → risoluzione minore, 32 el → maggiore)
  - Regioni (ROI: dx/sx, ventrale/dorsale): SÌ confrontabili — sono frazioni della mesh normalizzata, esistono identiche su entrambe
  - Indici clinici (GI, CoV, tidal variation regionale): SÌ confrontabili — adimensionali, normalizzati, non dipendono dalla risoluzione

### Domande di Riccardo
- "pyEIT non lo considera se Timpel ha 32 elettrodi?" → pyEIT è agnostico — configuri tu il protocollo con `n_el=32`. Il `vv` ha tante colonne quante il protocollo definisce.
- "se non so giustificare tutti i parametri non è un problema?" → Con GREIT puoi citare il consensus paper — i parametri sono validati dalla comunità. Non devi derivarli da zero.
- "il torace non è sempre circolare considerato?" → Sì, sempre — sia Dräger che Timpel. Avevo sbagliato a differenziarli su questo punto.
- "le features che tiriamo fuori sono confrontabili?" → Le ROI e gli indici clinici sì. I pixel raw no.

### Epifanie e insight
- "Ah quindi il .bin ha già le immagini ricostruite da Dräger — non ho bisogno di pyEIT per le feature cliniche. pyEIT lo uso per validare e per la pipeline open source cross-vendor!"
- "Se dimostro che GREIT ≈ algoritmo Dräger proprietario, ho una pipeline standardizzata che funziona su qualsiasi vendor. Un medico periferico carica il file, il centrale confronta tutti con lo stesso algoritmo validato."
- "Non confronto pixel per pixel tra Dräger e Timpel — confronto GI, CoV, tidal variation per ROI. Quelli sono adimensionali e la risoluzione non conta."

### Obiettivo scientifico esplicito (da salvare in roadmap)
La contribuzione scientifica centrale di fastEIT:
1. Parsare .eit (Dräger) e .x (Timpel) — nessuno ce l'ha open source
2. Ricostruire con GREIT (Adler 2009) — algoritmo standardizzato e citabile
3. Validare GREIT vs .bin Dräger (gold standard CE marcato) → dimostrare equivalenza
4. Applicare stessa pipeline a Timpel → cross-vendor comparison possibile
5. Risultato: **primo software open source per confronto EIT multi-vendor con pipeline standardizzata e validata**

Potenziale paper: *"Open-source vendor-agnostic EIT reconstruction pipeline: validation against PulmoVista 500 proprietary algorithm and cross-vendor comparison"*

### Note tecniche utili per dopo
- Pipeline clinica primaria: `.bin` → pixel Dräger → feature (nessuna ricostruzione richiesta)
- Bridge pyEIT: `.eit` → `vv` (Task 2.4.2) → GREIT → immagini proprie → confronto con `.bin`
- `metadata["n_electrodes"]` e `metadata["n_measurements"]` in `RawImpedanceData` servono esattamente per configurare il protocollo pyEIT automaticamente nel bridge

---

## Sessione 2026-03-25 — Reverse engineering .eit format + notebook

### Task lavorate
- Revisione e aggiornamento CSV task (17+ task da in_progress/todo → done)
- Reverse engineering completo formato `.eit` PulmoVista 500
- Scrittura notebook `docs/reverse_eng/eit_format.ipynb` da zero
- PR #18 mergeata su `dev`

### Concetti spiegati

- **Layout binario frame .eit**: ogni frame è 5495 bytes. Il blocco signal è 600 float64 (da byte 16 a 4816). Contiene trans_A[208], gruppi unknown, injection_current[16], voltage_A/B[16], trans_B[208], altri campi. Non 600 valori "casuali" — ogni sottogruppo ha un significato fisico preciso.
- **float64 vs float32**: l'errore classico nel reverse engineering binario. Se leggi 8 byte come due float32 invece di un float64 ottieni valori completamente sbagliati (es. -3.4e12 che sembrava un sentinel era un artefatto). La regola: prima testa sempre entrambi i tipi e confronta con i valori attesi fisicamente.
- **Calibrazione EIDORS**: `vv = ft[0]*trans_A - ft[1]*trans_B` dove `ft = [0.00098242, 0.00019607]`. Costanti empiriche stimate dagli sviluppatori EIDORS nel 2016 correlando con output `.get`. Non derivate da specifica hardware — "best possible guess". Riccardo ha imparato che in fisica clinica la calibrazione è spesso empirica, non analitica.
- **injection_current**: `vvOrig(225:240)` in MATLAB = frame bytes [1808:1936]. Dividendo per `fc=194326.3536` ottieni Ampere. È la misura real-time della corrente effettivamente iniettata (non quella nominale di 5mA).
- **voltage_A/B differenziale**: `(voltage_A - voltage_B) / fv` dove `fv=0.11771`. La differenza tra i due set annulla il rumore di modo comune. `fv` anche questo empirico.
- **mk_stim_patterns**: funzione EIDORS/FEM che descrive il protocollo di stimolazione (dove inietti, dove misuri). NON la implementiamo noi — pyEIT ha il suo equivalente. Noi passiamo i dati `vv` calibrati a pyEIT e diciamo "adjacent drive, 16 elettrodi".
- **Patient data redaction**: prima di stampare l'header ASCII, la funzione `redact_header()` sostituisce i valori di campi sensibili (`Filename`, `Name`, `Patient ID`, `Age/DoB`, `Short Comment`) con `[REDACTED]`. Il file originale non viene mai toccato.

### Domande di Riccardo
- "gli altri dati li buttiamo?" → No. `injection_current`, `voltage_A/B`, `gugus`, `unknown_*` vanno tutti in `aux_signals` — i named in campi espliciti, i truly-unknown raw per ricercatori futuri.
- "mk_stim_patterns dobbiamo implementarlo noi?" → No. pyEIT lo gestisce. Noi organizziamo i dati nel formato che si aspetta.

### Epifanie e insight
- "Ah quindi -3.4e12 non era un sentinel, era float32 sbagliato che leggeva byte di float64!"
- "Il commento `# estimated AA: 2016-04-07` di EIDORS significa che anche loro non sapevano esattamente — stavano indovinando con buona correlazione"
- "Il PulmoVista misura sia la corrente che la tensione reale ogni frame — non fida solo del nominale 5mA"

### Note tecniche
- `sep_offset` nell'header: l'offset assoluto dal byte 0 del file dove si trova il separatore `b'**\r\n\r\n\r\n'`. `binary_start = sep_offset + 8`.
- `parse_header(raw)` restituisce `(format_version, sep_offset, binary_start, fields_dict)` — il terzo valore è già `sep_offset + 8`.
- HealthyLung.eit è V1.00 a 20Hz — NON usarla come esempio primario. patient01.eit è V1.30 a 50Hz, più rappresentativa del clinico.
- `redact_header()` usa `line.find(':')` — funziona anche per `Filename:` che non ha padding.
- Framerate **non** è fisso a 50Hz — va parsato da `Framerate [Hz]` nell'header. HealthyLung è 20Hz.
- CSV task corrotto da script di aggiornamento (apertura in write mode prima del controllo). Recuperato 236/257 task dalla trascrizione sessione precedente. Mancanti: 7 task bloccate Fase 8/9 + alcune Fase 4-6 — da reinserire quando arriva il momento.

---

## Sessione 2026-03-22 (Parte 2) — Implementazione TimpelTabularParser

### Task lavorate
- Implementazione completa `TimpelTabularParser` + `TIMPEL_FRAME_DTYPE` + `TIMPEL_AUX_FIELDS`
- 41 test verdi per Timpel (dtypes + parser)
- Rimozione lazy import: Timpel ora tracciato git, import diretto
- Fix code-review: F1 (docstring timestamps), F2 (sentinel threshold), W3 (InvalidSliceError), W7 (file_format da extension)
- Refactor docs: `data_model.md` senza routing, `parsing_layer.md` con file layout
- 147 test totali, coverage 93%

### Concetti spiegati
- **Formato Timpel CSV**: 1030 colonne per riga, senza header, delimiter virgola. Cols 0-1023 = pixel 32×32 (row-major), cols 1024-1029 = aux signals (airway_pressure, flow, volume, min_flag, max_flag, qrs_flag). Sentinel −1000.0 = canale disconnesso → NaN.
- **np.loadtxt vs np.fromfile**: Timpel è testo CSV → `np.loadtxt` con `delimiter=","`. Dräger .bin è binario → `np.memmap` con `dtype` strutturato. Stessa architettura, input diversissimi.
- **Timestamp sintetici**: Timpel non ha colonna timestamp → generiamo noi: `ts[i] = (first_frame + i) / 50.0`. Questo preserva l'indice assoluto del frame anche quando si carica uno slice. Convention di eitprocessing.
- **TIMPEL_FRAME_DTYPE**: deve avere i campi `ts` (float64) e `pixels` (float32, 32×32) per essere compatibile con `ReconstructedFrameData.frames`. La struttura diversa da Dräger (che ha anche min_max_flag, event_marker ecc) è gestita con field opzionali nel data model.
- **Sentinel threshold comparison**: usare `values < TIMPEL_NAN_SENTINEL + 1.0` invece di `values == float32(sentinel)` perché −1000.0 è esattamente rappresentabile ma altri sentinels futuri potrebbero non esserlo. È più robusto di una equality comparison, anche se oggi funzionerebbe uguale.
- **Convenzione timestamp multi-vendor**: Dräger salva il tempo come frazione di giorno (0.0–1.0 = wall clock). Timpel non ha timestamp → secondi dall'inizio registrazione. Documentato in docstring `timestamps` property e in `data_model.md`.

### Epifanie e insight
- "Il lazy import `try: from .timpel import X except ImportError` era necessario quando il modulo era gitignored. Appena aggiungi il file al repo, puoi toglierlo — è rumore che nasconde errori reali."
- "L'errore `skiprows=100` su file con 5 righe restituisce array `(0,)` 1-D, non `(0, 1030)` 2-D — numpy non può inferire il numero di colonne da un file vuoto. Bisogna gestire `raw.ndim == 1 and raw.size == 0` prima di fare il reshape."
- "Il code reviewer ha trovato che `file_format='csv'` era hardcoded anche per file .txt e .asc — fix facile: `path.suffix.lower().lstrip('.')`."

### Domande di Riccardo
- "prima di pushare ricontrolla con i vari agents che tutto sia a posto la documentazione ecc. mi raccomando tra i due file md in docs elimina le ripetizioni. il file sui modelli non deve parlare della sequenza di loading ed espansione." → Refactoring doc: `data_model.md` descrive solo i container di dati; `parsing_layer.md` descrive il routing, detection, registry, e ora include anche il file layout (sezione 5 aggiunta).

### Note tecniche utili per dopo
- `np.loadtxt` con `skiprows=N` su file con <N righe → UserWarning + array `(0,)` vuoto, non eccezione
- Sentinel threshold: `< TIMPEL_NAN_SENTINEL + 1.0` = `< -999.0` — sicuro perché valori fisiologici sono sempre >0
- `TIMPEL_AUX_FIELDS` è una tuple ordinata: l'ordine corrisponde alle colonne 1024→1029 — non cambiare l'ordine
- Test validate() deve coprire anche .txt oltre che .csv — estensioni multiple = test multipli

---

## Sessione 2026-03-22

### Task lavorate
- Code quality review e fix post code-review dei tre subagent (eit-ci-checker, eit-doc-keeper, eit-code-reviewer)
- Espansione test suite: 55 → 99 test, coverage 70% → 92%
- Fix bug reale in bin_parser.py scoperto dai nuovi test
- Fix CI GitHub Actions (ruff format + test paziente gitignored)

### Concetti spiegati
- **memmap read-only copy bug**: `np.memmap[mode="r"][:]` crea una VIEW che eredita il flag read-only del mapping. La fix corretta è `.copy()` che alloca un nuovo array scrivibile. Il `[:]` sembra una copia ma non lo è.
- **ruff format vs ruff check**: la CI di GitHub esegue anche `ruff format --check` (non solo `ruff check`). Sono due cose diverse: `check` trova errori logici/stilistici, `format` controlla la formattazione Black-style. W191 (tabs) non è auto-fixabile con `--fix` — va riscritto il file.
- **Synthetic .bin file per test**: `np.zeros(n, dtype=FRAME_BASE_DTYPE).tofile(path)` crea un file binario valido senza dati paziente. I timestamps devono essere non-identici per la stima fs (tutti zero → `ValueError: Non-positive interval`).
- **BASE format ha Medibus**: entrambi BASE (4358 b) e EXT (4382 b) hanno `medibus_fields` popolato in `bin_formats.py` — `aux_signals` è sempre un dict, mai `None`. La distinzione reale è `has_pressure_pod_fields`.
- **skipif per test con file reali**: `pytest.mark.skipif(not Path("...").exists(), reason="...")` permette di avere test che girano solo in locale (dove esiste il file paziente) e vengono saltati su CI senza fallire.

### Domande e osservazioni di Riccardo
- "non so perche in locale con ambiente conda mi girano i test ma il push ha fallito entrambe le github action" → Causa 1: `ruff format --check` fallisce se non si esegue `ruff format` prima del commit. Causa 2: test che usano file paziente gitignored danno `FileNotFoundError` su GitHub CI.

### Epifanie e insight
- "Il bug del memmap non era ovvio: `[:]` sembra una copia ma è una view — solo `.copy()` rompe davvero il legame con il memmap read-only."
- "I test sintetici non richiedono dati paziente: basta costruire il dtype giusto con zeros, impostare timestamps validi, e scrivere su disco con `.tofile()`."
- "La CI di GitHub è più severa del check locale: controlla anche la formattazione, non solo i lint errors."

### Note tecniche utili per dopo
- Prima di ogni commit: `ruff check src/ tests/ && ruff format src/ tests/`
- `mapped_frames.copy()` non `mapped_frames[:]` per array scrivibile da memmap
- Coverage: 92% con 99 test — bin_parser, bin_utils, detection edge cases, error paths tutti coperti
- CHANGELOG.md rimosso dai file tracciati (non mantenuto in questa fase del progetto)

---

## Sessione 2026-03-21

### Task lavorate
- Task 1.2.2: implementazione `DragerBinParser.parse()` (memmap + slicing + sentinel handling + fs da timestamp)
- Task 3.3.1 (in progress): implementazione `DragerAscParser` nella nuova architettura parser

### Concetti spiegati
- **Parser architecture vendor-ready**: detection separata dal parser concreto. `detect_vendor_and_format()` decide vendor+estensione, `loader` istanzia parser da registry, `parse_safe()` applica validate+parse in modo uniforme.
- **Perché il parser ASC non deve usare read_csv diretto fino a EOF**: il file Drager `.asc` contiene blocchi multipli con schemi colonna diversi (immagini + tabelle). Se leggi tutto da "Tidal Variations" in poi, Pandas rompe quando incontra un blocco con 69 campi invece di 11. Soluzione robusta: estrarre solo la sezione con schema coerente.
- **Section-based parsing**: leggere il marker (`Tidal Variations`), prendere header tabellare, poi consumare righe finché il numero colonne resta costante e la prima colonna resta indice immagine numerico.

### Epifanie e insight
- "Il parser è molto più stabile se tratta l'ASC come documento a sezioni e non come CSV unico."
- "La stessa architettura ci permetterà di aggiungere Timpel senza toccare il core del routing."

### Note tecniche utili per dopo
- `detect_tabular_vendor()` ora riconosce Drager via header `DraegerEIT Software` e ha fallback euristico per Timpel.
- Nuovo container `TabularData` con `table: DataFrame`, `n_frames` e `duration` coerenti con il modello dati esistente.
- Test aggiunti per detection/parse/load su file reali ASC Drager (`patient01.asc`).
- Stato test dopo le modifiche: **57 passed**.

### Addendum naming refactor
- Rimossi gli alias `BinData` e `TabularData`: restano solo i nomi canonici `ReconstructedFrameData` e `ContinuousSignalData`.
- **Perché**: il nome della classe deve descrivere il contenuto dei dati, non l'estensione del file sorgente. Questo evita che un futuro export Timpel con matrici 32×32 finisca in una classe semanticamente "Drager-bin".
- Possibile evoluzione futura: introdurre un `TabularData` più astratto solo se servirà davvero come superclass generica per tabelle da ventilatore o export misti.

---

## Sessione 2026-03-20

### Task lavorate
- Fix architetturale `dtypes.py`: refactor MEDIBUS fields confermato da eitprocessing (Task 1.3.2 revisione)
- Aggiornamento `test_dtypes.py`: fixing semantica idx 51

### Concetti spiegati

- **Architettura MEDIBUS BASE vs EXT — la scoperta chiave**: I due formati `.bin` (4358 bytes BASE e 4382 bytes EXT) NON sono in relazione "BASE + 6 extra". Condividono i campi **idx 0-50** (51 campi comuni), ma **divergono completamente da idx 51**:
  - BASE (4358 bytes): idx 51 = `time_at_low_pressure` (Tlow BiLevel) — è l'unico campo aggiunto dopo il comune
  - EXT (4382 bytes): idx 51 = `high_pressure` (PHigh), idx 52 = `low_pressure` (Plow), idx 53 = `time_at_low_pressure` (Tlow — spostato da 51!), idx 54-57 = Pod channels
  - **Tlow si sposta da idx 51 (BASE) a idx 53 (EXT)** — questo è il punto più sottile e la fonte del bug precedente
  - Fonte: eitprocessing `draeger.py` (Apache-2.0) definisce `_bin_file_formats` con due dizionari completamente separati

- **Perché il precedente "fix" era sbagliato**: In una sessione precedente avevamo usato il file `.txt` del software Dräger (che mostra 58 canali in 1-indexed) come guida. Abbiamo messo PHigh a idx 51 in `MEDIBUS_FIELDS`. Ma `MEDIBUS_FIELDS` dovrebbe rappresentare il formato BASE (4358 bytes) dove idx 51 è **Tlow**, non PHigh. PHigh appare solo nel formato EXT e solo lì.

- **Struttura corretta in `dtypes.py`**:
  ```python
  _MEDIBUS_COMMON        # idx 0-50, 51 campi, identici in entrambi
  MEDIBUS_BASE_FIELDS    # = COMMON + [Tlow@51]  → 52 campi
  MEDIBUS_EXT_FIELDS     # = COMMON + [PHigh@51, Plow@52, Tlow@53, Pod×4@54-57] → 58 campi
  MEDIBUS_FIELDS         # alias di MEDIBUS_BASE_FIELDS (backward compat)
  MEDIBUS_BASE_INDEX     # dict name→index per BASE
  MEDIBUS_EXT_INDEX      # dict name→index per EXT
  MEDIBUS_INDEX          # alias di MEDIBUS_BASE_INDEX (backward compat)
  ```

### Epifanie e insight
- "Ah quindi il `.txt` con 58 canali del software Dräger rappresenta il formato EXT, non il BASE. Quando vai a idx 52 nel software vedi PHigh perché stai guardando un file con PressurePod connesso."
- "Il modo giusto per leggere eitprocessing era guardare la struttura di dati `_bin_file_formats` — due dizionari separati, non uno derivato dall'altro."

### Domande e risposte
- "Perché Tlow è presente sia in BASE che in EXT se in BASE non puoi settare BiLevel?" → Tlow è inviato dal ventilatore anche in ventilazione convenzionale, con valore sentinel `-1000.0`. Il Dräger include sempre il campo; se la modalità BiLevel non è attiva manda -1000.0.

### Note tecniche utili per dopo
- `high_pressure` esiste **solo** in `MEDIBUS_EXT_INDEX`, non in `MEDIBUS_BASE_INDEX`. Se provi `MEDIBUS_INDEX["high_pressure"]` ottieni `KeyError`. Usare sempre il dict giusto per il formato del file.
- I sentinel: `-1000.0` = parametro non applicabile/modalità non attiva; `0xFF7FC99E` = hardware non connesso (Pod assente)
- 22 test verdi dopo il fix — tutti i test semantici ora riflettono la struttura corretta

---

## Sessione 2026-03-19 (sera)

### Task lavorate
- Fix Streamlit taskboard: auto-reload on CSV mtime change
- Fix PRIORITY_VALUES: aggiunto "critical" a data.py
- Pianificazione task 1.1.1 (detect_frame_dtype) e 1.2.2 (parse_bin)

### Concetti spiegati

- **Streamlit file-watching**: Streamlit fa watching solo sui file `.py`, non sui `.csv`. Per rilevare modifiche esterne al CSV basta controllare `mtime` (modification time) ad ogni rerun — se cambia, cache miss automatico.
- **mtime come cache key**: passare `mtime: float` a `@st.cache_data` trasforma il timestamp in chiave di cache. Se Claude Code scrive il CSV, il prossimo rerun vede un `mtime` diverso → nuova chiamata → ricarica dal disco. Elegante e senza polling.

### Note tecniche
- `Path.stat().st_mtime` restituisce un float (secondi da epoch) — si può usare direttamente come chiave per `@st.cache_data`
- `st.session_state.get("csv_mtime")` ritorna `None` alla prima esecuzione → diverge da `_mtime` → primo caricamento automatico
- Il pulsante "🔄 Ricarica dal CSV" ora fa solo `st.cache_data.clear() + st.rerun()` — il mtime check fa il resto

---

## Sessione 2026-03-19 (pomeriggio)

### Task lavorate
- Fix MEDIBUS field ordering in dtypes.py (Task 1.3.2 + 1.3.5 follow-up)
- Estensione notebook bin_format.ipynb con sezione Healthy Lung (Task 1.4.6)

### Nuovi file di test
- `01_Healthy_Lung_01.bin/.eit/.asc` — registrazione 1118 frame, 20 Hz, NO PressurePod, NO MEDIBUS
- `elencoMEDIBUSdatafromDragerSoftware.txt` — lista manuale 58 canali Dräger

### Concetti spiegati

- **Ordinamento canali MEDIBUS**: Il channel list del software Dräger è la fonte più diretta di verità. PHigh (peak BiLevel pressure) è il canale 51 (0-indexed) = l'ultimo dei 52 canali standard, non un campo "extra PressurePod". I campi extra (idx 52-57) sono: Plow, Tlow, ~Paw, ~Pes, ~Ptp, ~Pgas.

- **Sentinelle MEDIBUS**: Due valori speciali distinti:
  - `0xFF7FC99E` = -3.4e38 → hardware NON connesso (MEDIBUS cable absent)
  - `0xC47A0000` = -1000.0 → connesso ma parametro non applicabile o nessun respiro calcolato ancora (es. Plow/Tlow in modalità convenzionale non-BiLevel)

- **BiLevel vs ventilazione convenzionale**: PHigh, Plow, Tlow sono parametri della modalità BiLevel (CPAP bifasica). In ventilazione convenzionale normale, questi hanno valore -1000.0 (sentinel "N/A"). Importante per l'interpretazione clinica: non confondere -1000.0 con un valore reale di pressione!

### Epifanie e insight

- **"Il Dräger usa SEMPRE 4382 byte/frame!"** — Anche una registrazione senza PressurePod ha frame da 4382 byte. La differenza non è nella struttura del frame ma nei valori: Pod assente = idx 54-57 hanno `0xFF7FC99E` (-3.4e38). Questo semplifica il parser: non serve distinguere BASE/EXT prima di aprire il file, basta guardare il sentinel a idx 54.

- **"Il .bin e il .asc non partono dallo stesso istante!"** — Il file Healthy_Lung ha un offset di +8 secondi tra il timestamp del .bin (frame 0) e il timestamp del primo frame esportato nel .asc. Il .asc è generato dal .eit, non dal .bin. I due file coprono periodi leggermente diversi.

- **"Healthy Lung è a 20 Hz, non 50 Hz!"** — dt = 0.05s invece di 0.02s. Il PulmoVista supporta framerate configurabili. Il parser dovrà rilevare il framerate dal timestamp, non assumere sempre 50 Hz.

- **"I pixel del .asc non sono comparabili direttamente con il .bin"** — Il .asc applica: (1) sottrazione della baseline (media dell'intera registrazione), (2) filtro low-pass 50 bpm. I pixel grezzi del .bin sono valori assoluti. Per cross-validare correttamente bisognerebbe applicare gli stessi preprocessing.

### Domande aperte
- PHigh a idx 51 vale -1000.0 in tutti i file (anche patient01 con MEDIBUS connesso). Quand'è che PHigh diventa non-zero? Forse solo in modalità BiLevel con MEDIBUS configurato per esportarlo?
- Il formato BASE (4358 byte) esiste davvero su qualche device o firmware? Non ne abbiamo ancora visto uno.

### Note tecniche
- `np.memmap(..., dtype=FRAME_EXT_DTYPE)` funziona uguale su file con e senza PressurePod — distinzione solo nei valori, non nella struttura
- Per rilevare PressurePod: `frames['medibus_data'][:5, 54].view(np.uint32) != 0xFF7FC99E`
- Per rilevare MEDIBUS: `frames['medibus_data'][0, 0].view(np.uint32) != 0xFF7FC99E`
- Per rilevare framerate: `np.diff(frames['ts']).mean() * 86400` → secondi per frame

---

## Sessione 2026-03-17 (Sessione 0 — Pianificazione)

### Task lavorate
- Nessuna task di codice. Sessione di pianificazione, ricerca landscape, architettura, roadmap.

### Concetti appresi (davvero capiti)

- **File .bin e .eit sono due file separati**: il PulmoVista li genera in parallelo. Il .bin ha le immagini ricostruite 32×32, il .eit ha i dati grezzi dagli elettrodi. Per il mio uso clinico quotidiano basta il .bin.

- **Il formato .eit Dräger NON è quello che pensavamo**: i blocchi tipo 3/7/8/10 sono del formato Carefusion (altro produttore, stessa estensione file). Il formato Dräger ha header testuale + dati binari. Scoperto analizzando il codice EIDORS.

- **208 misure, non 256**: il PulmoVista con 16 elettrodi produce 208 misure indipendenti per frame (16 iniezioni × 13 misure, escludendo auto-misure).

- **EIDORS è GPL**: significa che non possiamo copiare il loro codice nel nostro progetto Apache-2.0. Ma possiamo leggere il codice per capire il formato del file, e poi scrivere il nostro parser da zero. Le specifiche di un formato file non sono copyrightabili, il codice sì.

- **Il valore del progetto**: non è nel preprocessing (che reimplementiamo da eitprocessing/letteratura) ma nel pipeline end-to-end: file → features → database breath-level. Nessuno ce l'ha.

- **PEEP step detection è un buco nel campo**: nessun tool fa detection automatica dei cambi PEEP. Né eitprocessing, né nessun altro.

- **pyEIT produce una mesh, non 32×32**: serve interpolazione per rimappare a 32×32. È lo stesso passaggio che fa il PulmoVista internamente.

### Concetti visti ma DA STUDIARE a fondo (non li so usare)

- **struct.unpack**: so che serve per leggere byte e interpretarli come numeri, ma devo fare esercizi pratici
- **numpy.fromfile e numpy.frombuffer**: so che esistono e che sono più veloci di struct per array grandi, ma non li ho mai usati
- **numpy dtype strutturato**: concetto potente (definisci il layout di un frame intero), ma non l'ho mai provato
- **Endianness**: so che il PulmoVista è little-endian e che si specifica con '<', ma devo fare pratica
- **mmap (memory-mapped files)**: ho solo letto che esiste, non ho idea di come si usa in pratica
- **scipy.signal**: find_peaks, butter, sosfiltfilt — so cosa fanno concettualmente ma non li ho mai usati
- **Watershed, Otsu**: metodi per lung mask — concetti vaghi, da studiare quando arrivo alla Fase 4
- **HDF5 / h5py**: so che è per salvare matrici grandi, mai usato

### Epifanie

- "Il .eit e il .bin sono due file separati generati in parallelo dal PulmoVista durante la stessa registrazione — non uno dentro l'altro!"

- "eitprocessing non fa PEEP step detection automatica. Nessuno lo fa. È un buco enorme nel campo."

- "Il parser a blocchi tipo 3/7/8/10 che avevamo analizzato è il formato CAREFUSION, non Dräger! Il formato Dräger è completamente diverso. Scoperto analizzando il codice EIDORS completo."

- "pyEIT non produce 32×32 ma una mesh triangolare. Serve interpolazione (scipy griddata) per rimappare a 32×32 e rientrare nel pipeline. È lo stesso passaggio che fa il PulmoVista internamente."

- "Il valore del progetto non sta nel preprocessing (copiato/reimplementato da eitprocessing) ma nel pipeline end-to-end: file → preprocessing → features → database breath-level → export per R/ML. Nessuno ce l'ha."

- "Le correzioni manuali dell'utente nella GUI (mask, breath detection, PEEP step) diventano training data per ML futuro. Flywheel: più usi → più dati → migliori predizioni → meno correzioni."

### Note da ricordare per dopo
- Il consensus TREND 2017 ha 193 pagine di supplementi — leggere almeno le definizioni dei parametri
- I fattori di scala EIDORS per le transimpedenze sono STIMATI ("estimated AA: 2016-04-07") — vanno verificati confrontando output Python vs EIDORS sullo stesso file
- eitprocessing è Apache-2.0 (posso studiare il codice), EIDORS è GPL (posso leggere per capire il formato, NON copiare codice)
- Il PulmoVista usa 16 elettrodi, pattern adjacent-drive, 5mA iniezione
- Conversazione 1 con Claude ha spiegazioni dettagliate di struct/numpy/endianness — rileggere quando serve

---

## Sessione 2026-03-18 (Sessione 2 — Task Board UI + Riorganizzazione CSV)

### Task lavorate
- Nessuna task di codice clinico. Sessione di tooling e riorganizzazione.
- Costruito **Task Board Streamlit** (`tools/taskboard/`) per gestire le task senza editare il CSV a mano
- Rianalisi completa del CSV e riorganizzazione dipendenze/priorità

### Cosa è stato costruito

**Task Board Streamlit** — `tools/taskboard/app.py` + `tools/taskboard/data.py`
- Tabella editabile con `st.data_editor`: colonne, selectbox per status/priority, modifica inline
- Colonna `_dot` con emoji colorate per status (🟢 done, 🟠 in_progress, ⬜ todo, 🔴 blocked)
- Tab Progress: metriche globali + barre avanzamento per ogni fase
- Tab Add Task: form con assegnazione automatica task_id (max+1)
- Si lancia con: `cd tools/taskboard && streamlit run app.py`

### Riorganizzazione CSV

**VIZ anticipate**: plot di debug spostati nelle fasi in cui servono, non alla fine
- V.1 → Fase 1 (debug parse_bin), V.2/V.4/V.5 → Fase 4 (debug lung mask + breath detection)

**Fasi 8/9 aggiunte al CSV come blocked**:
- Fase 8 = R wrapper (subito dopo la lib, depends_on fase 6)
- Fase 9 = Django webapp (più avanti, depends_on fase 6)

**Scope chiarito**: solo ventilazione, no perfusione
- 4.3.1 filtro perfusione → BACKLOG, B.3 analisi perfusione → eliminata

**Priorità**: task_id 86 (Flask backend) → critical, task_id 8 (optional deps) → low

### Stato CSV finale
- **211 task** totali | 12 done | 5.7% completato

### Epifanie
- "I plot di debug non sono opzionali — V.1 dopo parse_bin e V.2 dopo lung_mask accelerano la verifica di 10x rispetto a print() su array."
- "Il wrapper R viene prima del Django perché uso R ogni giorno. La GUI per i clinici è utile ma non è la mia pipeline principale."

---

## Sessione 2026-03-17 (Sessione 1 — Orientamento sistema + Analisi CSV + Kickoff Fase 0)

### Task lavorate
- Nessuna task di codice. Sessione di orientamento: architettura del sistema, analisi incongruenze CSV, pianificazione Fase 0.

### Come funziona il sistema (capito oggi)

**Architettura a 3 livelli**:
1. **CLAUDE.md** — caricato automaticamente ad ogni sessione. Contiene identità, regole ferme, flusso di lavoro, path dei file. Claude lo legge prima di rispondere a qualsiasi cosa.
2. **Memory system** (`~/.claude/projects/.../memory/`) — persistenza tra sessioni: chi sei, come vuoi collaborare, regole apprese. Caricato automaticamente come contesto.
3. **File di progetto** (`.claude/*.md`, `.claude/*.csv`) — da leggere attivamente all'inizio sessione. tasks.csv = stato progetto, roadmap = istruzioni, reference = paper+formule, learning_journal = questo diario.

**I 3 agent specializzati** (`.claude/agents/`):
- `eit-ci-checker` — prima di ogni commit: ruff lint, pytest, coverage 80%
- `eit-code-reviewer` — dopo aver scritto codice: correttezza scientifica, citazioni, test
- `eit-doc-keeper` — dopo ogni task completata: README, docstring, docs/ coerenti

**Flusso tipico sessione**: Claude legge CSV → presenta task consigliate → spiega teoria → esempio minimo → tu scrivi nell'IDE → agent review → Claude aggiorna CSV e journal.

### Analisi incongruenze CSV (fatto oggi)

**3 fix applicati al CSV**:
- Task 15 (PeepStep): dipende da RawData(13) + Breath(14), non solo da 13 — PeepStep contiene `breath_indices`
- Task 16 (Study): dipende da 13+14+15, non solo da 13 — Study aggrega Breath e PeepStep
- Task 155 (PEEP detection Medibus): dipende da 30 (Medibus extraction) + 118 (breath detection), non solo da 30 — non puoi fare PEEP detection su singoli breath se non hai ancora rilevato i breath

**Task 50 (Download EIDORS)**: priority era già `critical` nel CSV, nessun fix necessario.

### Stato progetto attuale
- 205 task | 0 done | 0 in_progress | 0 blocked
- Fase 0: 23 task (infrastructure + data model) — tutto todo, nessuna dipendenza esterna
- Percorso critico realistico: **0 → (1+2 in parallelo) → 4 → 5 → 6**

### Rischi operativi identificati
1. **File reali (.eit e .bin)**: task 32 e 56 sono `critical` e bloccanti per validazione Fasi 1+2. Da procurare prima possibile.
2. **EIDORS study**: richiede scaricare e leggere codice MATLAB — tempo reale di lettura da parte mia.
3. **Scope creep Fase 5**: 41 task, alcune già marcate post-v1 — tenere la barra.

### Prossime task consigliate (sbloccate ora, dipendenze vuote)
| Task | Titolo | h | Perché prima |
|---|---|---|---|
| 0.2.1 | Creare repo GitHub | 0.5h | Sblocca intera struttura |
| 0.2.2 | Struttura directory | 1h | Dipende da 0.2.1 |
| 0.7.2 | LICENSE Apache-2.0 | 0.2h | Dipende da 0.2.1, rapidissima |
| 0.1.1 | Studio struct/numpy binari | 2h | Base per tutto il parsing |
| 0.1.2 | Studio scipy signal processing | 2h | Base per preprocessing |

**Sequenza consigliata**: `0.2.1 → 0.2.2 → 0.7.2` (infrastruttura) poi `0.1.1 + 0.1.2` (studio teorico).

### Obiettivo fine Fase 0 (checklist)
- [ ] `src/eitclinical/` con moduli vuoti + `__init__.py`
- [ ] `pyproject.toml` installabile con `pip install -e .[dev]`
- [ ] `pytest tests/` passa (0 test, ma senza errori di import)
- [ ] `ruff check src/` pulito
- [ ] `BaseParser`, `RawData`, `Breath`, `Config` esistono con type hints

---

### Aggiornamento fine sessione — File reali confermati disponibili

**File Dräger reali già in mano** (confermato 2026-03-17):
- `.eit` ✅, `.bin` ✅, `.txt` ✅
- Varianti con e senza PressurePod ✅
- Più registrazioni diverse ✅

**Task 32 (1.4.1) e 56 (2.1.1) marcate `done`** — erano le due `critical` che sembravano bloccanti. Con i file reali già disponibili:
- Validazione Fase 1 (parser .bin) → sbloccata
- Validazione Fase 2 (parser .eit) → sbloccata

**Rischio operativo #1 eliminato**: "Procurarsi file reali" era il rischio #1 identificato nella Sessione 0. Non è più un rischio — i file ci sono.

**Prossime task (sessione dopo)**: `0.2.1 → 0.2.2 → 0.7.2 → 0.1.1 → 0.1.2`

---

### Aggiornamento — Rename eit_clinical → fastEIT + Scaffold Fase 0

**Rename completato** (2026-03-17): tutto il progetto ora usa `fastEIT` come nome. Il package Python si importa come `import fasteit` (minuscolo per convenzione PEP 8).

**Task completate in questa sessione**:
| Task | Titolo | Note |
|---|---|---|
| 0.2.1 | Creare repo GitHub | Già presente (retroattivo) |
| 0.7.2 | LICENSE Apache-2.0 | Già presente (retroattivo) |
| 0.2.2 | Struttura directory | `src/fasteit/` con tutti i moduli placeholder |
| 0.2.3 | docs e notebooks | `docs/eit_format.md`, `docs/eidors_analysis.md`, `docs/reverse_eng/`, `notebooks/` |
| 0.3.1 | pyproject.toml | hatchling, deps core + [pyeit] + [dev], pytest markers, ruff |
| 0.7.1 | README base | Badge + differenziatori + installazione + disclaimer |

**Cosa è il pyproject.toml** (spiegazione rimandata — vedi prossima sessione)

**Prossime task sbloccate**:
- `0.3.2` — optional dependencies [pyeit] e [dev] (già fatto in 0.3.1, da valutare se separare)
- `0.4.1` — GitHub Actions workflow test.yml
- `0.4.2` — workflow lint.yml
- `0.5.1` — BaseParser classe astratta
- `0.1.1 / 0.1.2` — studio struct/numpy + scipy (teoria, nessuna dipendenza)

---

### Sessione teorica — Python packaging, GitHub Actions, binary parsing, signal processing

#### Python packaging (approfondimento)
- **`import` in Python**: cerca `fasteit` in `sys.path` — lista di cartelle dove Python guarda. `src/` layout obbliga a usare il pacchetto installato, non i file locali grezzi.
- **`__init__.py`**: trasforma una cartella in un modulo. Viene eseguito ad ogni `import fasteit`. Posto giusto per `__version__` e API pubblica.
- **`pip install -e .`**: editable install = collegamento simbolico a `src/fasteit/`. Modifica → subito visibile. Nessun reinstall necessario.
- **wheel vs sdist**: wheel = archivio già pronto (veloce). sdist = sorgente da compilare. Per numpy/scipy esistono wheel precompilate.
- **hatchling**: build backend moderno. Alternativa a setuptools senza `setup.py`.

#### GitHub Actions / CI/CD
- **CI**: ogni push → computer vergine → esegue i test automaticamente. "Vergine" = senza i tuoi pacchetti locali, variabili d'ambiente, file casuali.
- **CD**: se i test passano → pubblica/deploya automaticamente. Non ci serve ancora.
- **Struttura workflow**: `on:` (trigger) → `jobs:` → `steps:`. File YAML in `.github/workflows/`.
- **Runner**: VM Ubuntu fresca nel cloud GitHub. Distrutta dopo ogni run.
- **Matrix strategy**: ripete il job su più versioni Python in parallelo (`["3.10", "3.11", "3.12"]`).
- **`actions/checkout@v4`**: fa il git clone del repo nella VM. Primo step di ogni workflow.
- **`actions/setup-python@v5`**: installa la versione Python specificata.
- **`cache: pip`**: ricicla le dipendenze pip tra un run e l'altro se `pyproject.toml` non è cambiato.
- **workflow test (0.4.1)**: checkout → setup-python con matrix → pip install -e .[dev] → pytest
- **workflow lint (0.4.2)**: checkout → setup-python 3.12 → pip install ruff → ruff check + ruff format --check
- **Badge**: URL automatico `github.com/USER/REPO/actions/workflows/test.yml/badge.svg`

#### Letture assegnate — GitHub Actions
- GitHub Actions quickstart (docs ufficiale)
- Understanding workflows (docs ufficiale)
- `actions/setup-python` marketplace page (per parametri cache)
- Matrix strategy (docs ufficiale)

#### struct e numpy per file binari (0.1.1)
- **Byte raw**: i numeri nel .bin sono in memoria esattamente come in RAM. Nessun testo, nessun separatore.
- **Endianness**: little-endian = byte meno significativo prima. Il PulmoVista è little-endian. In numpy: `'<f4'` (il `<` è obbligatorio e non va dimenticato).
- **`struct.unpack(format, buffer)`**: legge un singolo valore da bytes. Utile per header con campi misti. Format string: `'<'` prefix + `'f'`/`'d'`/`'i'`/`'H'`/`'4s'` ecc.
- **`numpy.fromfile(file, dtype, count)`**: legge count valori in blocco. Velocissimo. Per leggere tutto il file di pixel in una riga.
- **`numpy.frombuffer(buffer, dtype)`**: come fromfile ma da bytes già in memoria.
- **numpy structured dtype**: descrive l'intero layout di un frame come un unico tipo (`np.dtype([('pixels', '<f4', (32,32)), ('timestamp', '<f8'), ...])`). Con questo leggi tutto il file in una riga e accedi ai campi per nome.
- **`numpy.memmap`**: file grandi → carica solo le pagine accedute. Non serve per file <500MB.

#### Letture assegnate — struct/numpy (0.1.1)
- `docs.python.org/3/library/struct.html` — sezione Format Characters
- `numpy.org/doc/stable/reference/arrays.dtypes.html` — Specifying and constructing data types
- `numpy.org/doc/stable/reference/generated/numpy.fromfile.html`
- `numpy.org/doc/stable/user/basics.rec.html` — Structured arrays

#### scipy signal processing (0.1.2)
- **Frequenze nel segnale EIT**: respiro 0.1-0.5 Hz, cardiaco 1.0-2.5 Hz. Zona morta 0.5-1.0 Hz che permette di separarli.
- **Filtro Butterworth**: risposta piatta nella banda passante, attenuazione monotona fuori. Parametri: cutoff (Hz), order (pendenza del taglio).
- **Zero-phase con `sosfiltfilt`**: applica il filtro avanti e indietro → i ritardi si annullano → nessuno sfasamento temporale. Fondamentale per non spostare i picchi respiratori.
- **`sos` vs `ba`**: `sos` (Second-Order Sections) è numericamente più stabile per filtri di ordine alto. Usare sempre `butter(..., output='sos')` + `sosfiltfilt`.
- **`find_peaks`**: trova massimi locali. Parametri chiave: `prominence` (quanto sporge il picco), `distance` (distanza minima tra picchi in campioni), `width` (larghezza minima).
- Per trovare valli (fine espirazione): `find_peaks(-signal, ...)`.

#### Letture assegnate — scipy (0.1.2)
- `docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html`
- `docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html`
- `docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html`
- `docs.scipy.org/doc/scipy/tutorial/signal.html` — sezione Filtering
- Real Python: "Signal Processing with Python and SciPy"

#### Prossima sessione — cosa fare
1. Leggere le pagine di documentazione indicate sopra
2. Scrivere `.github/workflows/test.yml` + `lint.yml` nell'IDE (task 0.4.1 + 0.4.2)
3. Fare push e vedere i workflow partire
4. Claude supervisiona i file e segnala problemi

---

### Convenzione learning journal (stabilita oggi)

**Regola ferma**: una sola entry per data (`## Sessione YYYY-MM-DD`). Se si lavora su più task nella stessa giornata, si **estende la stessa entry** — si aggiungono sezioni in fondo. Una nuova entry (`## Sessione YYYY-MM-DD`) si crea solo il giorno dopo.

---

## Sessione 2026-03-18

### Task lavorate
- 0.4.1: Workflow test.yml — creato da template GitHub, adattato per fastEIT
- 0.4.2: Workflow lint.yml — scritto da zero
- 0.4.3: pytest markers — già presenti in pyproject.toml (retroattivo)
- 0.3.2: optional dependencies — già incluse in 0.3.1 (retroattivo)
- Setup ambiente conda locale (fasteit, Python 3.11)
- Debug CI: fix cache pip path, fix pyproject.toml mancante su GitHub, fix src/ mancante
- Verifica coerenza documentazione con subagent doc-keeper e CI-checker

### Findings

**Inconsistenza principale corretta**: task 0.4.1 (test.yml), 0.4.2 (lint.yml) e 0.4.3 (pytest markers) erano marcate `todo` nel CSV ma il codice corrispondente esisteva già nel repo. Corrette a `done` con date 2026-03-17.

**CHANGELOG.md mancante**: il progetto non aveva un CHANGELOG. Richiede Write permission per creare il file — segnalato all'utente.

### Stato documentazione verificato

| File | Stato |
|---|---|
| README.md | Coerente con pyproject.toml — installazione e deps corretti |
| pyproject.toml | Coerente — hatchling, deps core + [pyeit] + [dev], pytest markers, ruff |
| src/fasteit/ placeholder | Tutti i file hanno docstring di modulo con TODO appropriati |
| docs/eit_format.md | Placeholder accettabile — contiene i known facts sul formato |
| docs/eidors_analysis.md | Placeholder accettabile — rimanda a Task 2.0.x |
| .claude/fastEIT_tasks.csv | Corretto — 0.4.1/0.4.2/0.4.3 ora marcate done |
| .github/workflows/ | test.yml e lint.yml presenti e coerenti con pyproject.toml |
| CHANGELOG.md | Mancante — da creare |

### Concetti appresi

- **conda vs venv**: conda gestisce anche la versione Python, non solo i pacchetti. Gli ambienti stanno in `~/miniconda3/envs/` — fuori dalla cartella progetto. Con venv l'ambiente è dentro il progetto (`.venv/`).
- **Workflow quotidiano conda**: `conda activate fasteit` → lavori → `conda deactivate`. Il nome dell'ambiente è una convenzione tua, conda non sa nulla del codice.
- **GitHub Actions — errori reali di oggi**:
  - `cache: pip` senza `cache-dependency-path` non trovava `pyproject.toml` → aggiunto path esplicito
  - `pyproject.toml` e `src/` non committati → CI non trovava il package da installare
  - Strategia commit incrementale: prima `pyproject.toml`, poi `src/fasteit/__init__.py`
- **Exit code 5 pytest**: pytest senza test restituisce exit code 5 (no tests collected) → CI fallisce. Soluzione: test placeholder con `test_import()`.
- **subagent workflow**: CI-checker e doc-keeper girano in background in parallelo, aggiornano CSV e journal autonomamente. Da usare sistematicamente.

### Note tecniche
- Il package si importa come `fasteit` (minuscolo), coerente con pyproject.toml `name = "fasteit"`
- `src/fasteit/__init__.py` espone solo `__version__` in `__all__` — corretto per fase scaffold
- `tests/test_placeholder.py` garantisce che CI non sia mai completamente vuoto
- La Quickstart nel README è già marcata "Coming soon" — nessuna promessa su API non implementata
- Ambiente conda `fasteit`: Python 3.11, tutti i deps installati, pytest + ruff verdi in locale

### Prossime task
- 0.1.1 + 0.1.2: studio teorico (struct/numpy + scipy) — letture assegnate sessione precedente
- 0.5.1: BaseParser classe astratta
- 0.5.2: RawData dataclass
- 0.5.3: Breath dataclass (da Flask)

---

### Sessione 2026-03-18 (continuazione) — Task 0.5.1 + 0.5.2: BaseParser + BaseData/BinData

#### Task lavorate
- **0.5.1**: `BaseParser` ABC — `src/fasteit/parsers/base.py`
- **0.5.2**: `BaseData` + `BinData` dataclass — `src/fasteit/models/base_data.py` + `bin_data.py`
- Fix bug dtype in `.claude/fastEIT_layerdata_class_architecture.md`

#### Concetti spiegati in questo piano

**ABC (Abstract Base Class)**:
- `from abc import ABC, abstractmethod` — crei un contratto che le sottoclassi DEVONO rispettare
- Se una sottoclasse non implementa un `@abstractmethod`, Python solleva `TypeError` all'istanziazione — non a runtime nel mezzo del codice, ma subito. È la differenza tra "scoprire il bug quando usi il parser" vs "scoprirlo quando lo definisci"
- `BaseParser` ha `parse()` e `validate()` come abstractmethod — qualsiasi parser concreto DEVE implementarli entrambi
- `parse_safe()` non è abstractmethod — è implementato in `BaseParser` e funziona per tutti i parser automaticamente (Template Method pattern)

**`dataclass` e `field(init=False)`**:
- `@dataclass` genera `__init__`, `__repr__`, `__eq__` automaticamente dai campi dichiarati
- `field(init=False, default=0)` — il campo NON va passato al costruttore. Python lo inizializza al default. Utile per campi calcolati.
- `field(default_factory=dict)` — per i campi mutabili (liste, dict): NON si scrive `metadata: dict = {}` perché quella dict sarebbe condivisa tra tutte le istanze! `default_factory` crea una nuova dict per ogni istanza.
- `__post_init__` — chiamato da Python subito dopo `__init__`. Posto giusto per calcoli derivati (es. `n_frames = len(frames)`).

**Gerarchia BaseData → BinData**:
- `BaseData` ha solo i campi comuni (filename, fs, metadata, n_frames, duration)
- `BinData(BaseData)` aggiunge i campi specifici del formato .bin (frames, medibus) e le properties
- `n_frames` e `duration` sono `init=False` in `BaseData` ma vengono settati da `BinData.__post_init__`

#### Bug corretto nel documento di architettura

**Problema**: `FRAME_BASE_DTYPE` aveva `minmax` (2×i4 = 8b) + `event_marker` (1×i4 = 4b) = 12 byte, ma il totale commentato diceva 4358. Il math non tornava: 4+4+4+4096+12+30+4+208 = 4362 ≠ 4358.

**Fix**: unificato in `minmax_event` (2×i4 = 8 byte). Rimosso `event_marker` separato.
Ora: 4+4+4+4096+8+30+4+208 = **4358 byte ✓**

Confermato da assert nel test: `assert _FRAME_BASE_DTYPE.itemsize == 4358`

#### Cosa è stato costruito

| File | Cosa fa |
|---|---|
| `src/fasteit/parsers/base.py` | `BaseParser` ABC con `parse()`, `validate()`, `parse_safe()` |
| `src/fasteit/models/base_data.py` | `BaseData` dataclass — campi comuni a tutti i formati |
| `src/fasteit/models/bin_data.py` | `BinData(BaseData)` — frames strutturati + properties (`timestamps`, `pixels`, `global_signal`, `roi_signals`, `roi_signal()`) |
| `tests/test_base_parser.py` | 6 test: TypeError su ABC, FileNotFoundError, ValueError, parse riuscito, coercizione str→Path |
| `tests/test_bin_data.py` | 14 test: n_frames, duration, shapes, valori, ROI, custom fs, errori |

**Tutti i test verdi: 21/21** (incluso il placeholder)

#### Note tecniche utili per dopo

- `dtypes.py` con `FRAME_BASE_DTYPE` ufficiale verrà scritto in Task 1.2.1. I test di ora usano un dtype inline identico nel fixture (così se il dtype cambia, i test del parser reale catturano la discrepanza).
- `BinData.frames` può essere `None` — `__post_init__` lo gestisce con `if self.frames is not None`. Questo permette di creare un `BinData()` vuoto senza errore (utile per test e future operazioni di merge).
- Properties come `timestamps` e `pixels` accedono direttamente al dtype strutturato numpy con `frames["fieldname"]` — nessuna copia, zero overhead.
- `roi_signal(roi)` valida con `roi not in range(4)` — più esplicito di `roi < 0 or roi > 3`.
- `from __future__ import annotations` in tutti i file — permette di usare type hints come stringhe forward reference senza import circolari.

#### Stato CSV aggiornato
- Task 0.5.1: ✅ done
- Task 0.5.2: ✅ done (rinominata: BaseData + BinData, non più RawData in session.py)

#### Prossima sessione
1. Studio eitprocessing — com'è parsato il .bin in Apache-2.0
2. Verifica con `openBins.m` del collega MATLAB
3. Task **1.1.1**: `autodetect_frame_size(file_size)` — scrivi + test con file reale
4. Task **1.2.1**: `FRAME_BASE_DTYPE` ufficiale in `src/fasteit/dtypes.py`

---

### Aggiornamento 2026-03-18 — Review letteratura EIT con GPT: 3 correzioni al progetto

Sessione di revisione critica della letteratura (Delphi 2025, Scaramuzzo 2024, review recenti). Nessun codice scritto, ma trovate 3 cose da correggere nel progetto:

**1. CoV mancava la direzione sinistra-destra**
Il Delphi 2025 cita esplicitamente CoV in entrambe le direzioni come misura standard. Noi avevamo solo la formula verticale (ventro-dorsale). Aggiunto:
- Formula CoV_LR: stessa struttura di CoV_VD ma sull'asse delle colonne
- Task 5.1.4 aggiornata: implementare entrambe (`cov_vd` e `cov_lr`)
- DB schema aggiornato con colonna `cov_lr`

**2. RVD formula era sbagliata: np.argmax ≠ t40 method**
Task 5.3.2 diceva `np.argmax` (tempo al picco). La formula standard da Muders 2012 è il **t40 method**: tempo in cui il pixel raggiunge il 40% del suo ΔZ tidale.
Sono cose diverse: argmax è il massimo assoluto, t40 è il punto a 40% della salita. t40 è più robusto perché non dipende dalla fase finale dell'inspirazione.
Implementazione: `np.argmax(cumulative_dz >= 0.4 * total_dz, axis=time_axis)`

**3. SDRVD è una metrica con nome proprio, non solo "RVD_inhomogeneity"**
La SD degli RVD pixel-wise è citata come SDRVD nel Delphi 2025 e nella Wisse survey. Nel reference doc la chiamavamo "RVD_inhomogeneity". Allineato il nome.

**Conferme dalla letteratura (nessuna modifica necessaria)**:
- La gerarchia frame → breath → step/patient è esattamente quella usata in letteratura. Buona scelta architetturale.
- TIV, EELI, GI, silent spaces, V/D ratio, ROI distribution, compliance regionale, ITV, OD/CL, pendelluft: tutti già nella roadmap con le priorità giuste (v1 vs post-v1).
- Delphi 2025 e Scaramuzzo 2024 confermano GI e CoV come standard per PEEP titration.
- Pendelluft e silent spaces: già in roadmap come post-v1 per clinical use diffuso ma non fondamentale v1.

---

## Sessione 2026-03-18 — Review documentazione post config.py + base_data.py

### Task lavorate
- Nessuna task di codice. Sessione di verifica documentazione (eit-doc-keeper).

### Bug trovati nei file modificati

**`src/fasteit/config.py` — 3 problemi**:

1. `PreprocessingConfig` e `AnalysisConfig` hanno il corpo vuoto — in Python un `@dataclass` senza campi né `pass` è un SyntaxError. Fix: aggiungere `pass` come corpo temporaneo.

2. `Config` usa `field(...)` come annotazione di tipo invece di come `default_factory`:
   ```python
   # SBAGLIATO (quello che c'è ora):
   device: field(default_factory=DeviceConfig)
   # CORRETTO:
   device: DeviceConfig = field(default_factory=DeviceConfig)
   ```

3. La docstring di modulo ha ancora banner TODO come se DeviceConfig non esistesse — va aggiornata.

**`src/fasteit/models/base_data.py` — 2 problemi**:

1. Import sbagliato: `from config import Config` fallisce nel src/ layout. Deve essere `from fasteit.config import Config`.

2. Default `fs` usa subscript invece di attributo: `Config.device[fs]` non esiste. Il problema più profondo è che `Config.device` è un `field(default_factory=...)` — un'istanza non esiste a livello di classe. Il valore di default di `fs` in `BaseData` dovrebbe essere un letterale (`50.0`) o prendere il valore da `DeviceConfig().fs`.

3. Il docstring dice `Default 20.0` per `fs` ma `DeviceConfig.fs = 50.0`. Il PulmoVista ha sampling rate 20 Hz per le immagini .bin, ma 50 Hz per i dati raw .eit — discrepanza da chiarire e documentare.

### Note su DeviceConfig.fs = 50.0 vs 20.0 Hz

- `DeviceConfig.fs = 50.0` — è il frame rate di campionamento interno del dispositivo
- Il .bin ha immagini ricostruite a 20 Hz (standard PulmoVista), non 50 Hz
- Il .eit ha i dati grezzi a 50 Hz
- `BaseData.fs` deve essere il fs effettivo del file letto, non un default fisso — impostarlo nella `__post_init__` del parser in base al formato
- La discrepanza tra docstring (20.0) e DeviceConfig (50.0) va risolta dichiarando esplicitamente i due valori come costanti separate

### Documentazione aggiornata in questa sessione
- CHANGELOG.md — aggiornato con Task 0.5.1, 0.5.2, 0.6.1 partial
- README.md — "Project status" aggiornato: scaffold complete, next = Fase 1
- fastEIT_tasks.csv — Task 0.6.1 marcata in_progress con note sui bug

### Prossime azioni prioritarie
1. Correggere i bug sintattici in `config.py` (aggiungere `pass` ai dataclass stub, correggere `Config` field annotations)
2. Correggere `base_data.py`: import path + default fs (usare letterale `50.0` o `DeviceConfig().fs`)
3. Chiarire e documentare la differenza tra fs .bin (20 Hz) e fs .eit (50 Hz)

---

## Sessione 2026-03-19 — Analisi comparativa .bin format + Task 1.2.1 dtypes.py

### Task lavorate
- Studio comparativo formato .bin: EIDORS (MATLAB) + eitprocessing (Python) + script MATLAB collega
- Task **1.2.1**: `src/fasteit/dtypes.py` — FRAME_BASE_DTYPE e FRAME_EXT_DTYPE definitivi
- Task **1.2.1b**: aggiornamento `BinData` — property timestamps, min_max_flags, event_markers
- Aggiornamento `fastEIT_layerdata_class_architecture.md` — dtype corretto

### Scoperta critica: ts è float64, non due float32

Il campo timestamp nei .bin è un singolo **float64 (8 byte)**, non due float32 (ts1, ts2).

Cosa memorizza il PulmoVista: `frazione di giorno` — un numero da 0.0 (mezzanotte) a 1.0
(quasi mezzanotte del giorno dopo). Per esempio 0.5 = mezzogiorno, 0.708... = 17:00.

**Per ottenere secondi**: `ts * 86400` (dove 86400 = secondi in un giorno).

**Perché EIDORS leggeva due float32**: MATLAB `fread(f, 1, 'float32')` legge 4 byte come float.
Chiamato due volte, legge i primi 4 byte (metà del double) e i secondi 4 byte (altra metà).
EIDORS usava `ts1` per il tempo — ma ts1 come float32 dei primi 4 byte di un float64 è un
valore numerico casuale, non un timestamp sensato! La rappresentazione interna IEEE 754 di
un double e di un float hanno bit di mantissa e esponente diversi.

**eitprocessing fa la cosa giusta**: `reader.float64() * 24 * 60 * 60` legge tutti gli 8 byte
come double e li moltiplica per 86400.

**Altra nota importante**: `np.unwrap(ts * 86400, period=86400)` gestisce il wraparound per
registrazioni che iniziano prima di mezzanotte e finiscono dopo. Questo va fatto nel BinParser,
non in BinData (che conserva i dati grezzi dal file).

### Struttura campi corretta (4358 byte base)

| Campo | Tipo | Bytes | Cosa è |
|---|---|---|---|
| ts | float64 | 8 | Frazione di giorno → ×86400 = secondi |
| dummy | float32 | 4 | Non usato |
| pixels | float32 (32,32) | 4096 | Immagine ricostruita C-order |
| min_max_flag | int32 | 4 | +1=picco insp, -1=valle esp, 0=niente |
| event_marker | int32 | 4 | Counter eventi (incrementa) |
| event_text | S30 | 30 | Testo evento ASCII |
| timing_error | int32 | 4 | 0=ok, non-zero=errore |
| medibus_data | float32 (52,) | 208 | 52 segnali Medibus |

Frame esteso (PressurePod, 4382 byte): medibus_data ha (58,) invece di (52,), +6 campi ×4b=+24b.

### I 52 campi Medibus — cosa sono

I primissimi 6 (idx 0-5) sono **waveform continue a 50 Hz**: pressure_airway, flow, volume, CO2 %.
Il resto (idx 6-51) sono **parametri aggregati dal ventilatore** aggiornati una volta per atto
respiratorio: PEEP, compliance dinamica, frequenza respiratoria, volumi espirati, ecc.

Per il nostro use case clinico i più importanti:
- **idx 14: PEEP** — fondamentale per Task 5.7.1 (PEEP detection Livello 1)
- **idx 0: airway_pressure** — pressione delle vie aeree in tempo reale
- **idx 1: flow** — flusso in tempo reale
- **idx 2: volume** — volume tidal in tempo reale

### Algoritmi clinici del collega MATLAB (da confrontare con 3 fonti a Fase 4/5)

Gli script del collega mostrano come lui implementa le stesse cose che noi faremo in Python.
**Li confronteremo su 3 fronti** quando arriveremo alle fasi relative: collega vs EIDORS vs
eitprocessing. Alcuni hanno già issue su GitHub che danno spunti utili.

**FindBreathsInAVector.m** → Task 4.10 (breath detection):
```matlab
smoothedTrace = smooth(scalarTrace);
[ValInsp, XpeakInsp] = findpeaks(smoothedTrace,
    'MinPeakWidth', fs/1.5,       % ≈33 frame = 0.67 secondi
    'MinPeakProminence', std(smoothedTrace));
```
Traduzione Python: `scipy.signal.find_peaks(smooth_signal, width=fs/1.5, prominence=np.std(signal))`
Il collega usa `smooth()` (media mobile MATLAB) come pre-filtro. eitprocessing non lo fa.
Noi useremo un butterworth bandpass prima di find_peaks (Task 4.10.1) — miglioramento rispetto
a entrambi.

**CalculateEELIandVT.m** → Task 5.x (features):
```matlab
eeli = median(exp_val);               % EELI = mediana dei valori espiratori
deltaZ = median(insp_val) - eeli;     % deltaZ ~ volume tidale
```
Semplice e robusto. La mediana invece della media rende EELI robusto agli outlier.

**createROIMasks.m** → Task 4.8 (ROI):
```matlab
mask.FourROIs.ROI1 = [ones(8, 32); zeros(24, 32)];  % righe 0-7 = ventrale
mask.FourROIs.ROI4 = [zeros(24, 32); ones(8, 32)];  % righe 24-31 = dorsale
```
Questo conferma che il nostro `BinData.roi_signals` è orientato correttamente.
In più, il collega ha `FourbyFourROIs` (4 ROI V/D × 2 lato sx/dx = 8 ROI totali) — utile
per Task 4.8 quando implementeremo le ROI avanzate.
Convenzione: colonne 0-15 = DESTRA paziente, colonne 16-31 = SINISTRA paziente.

**createSummersROIs.m** → Task 4.8.2 (ROI adattive Summers):
ROI adattate ai limiti reali del polmone nel frame (non strisce fisse 8 righe). Più preciso
per pazienti con polmoni piccoli o asimmetrici. Da confrontare con eitprocessing.

**applyMask.m** → Task 4.4/4.8 (lung mask):
Applica una maschera binaria 32×32 al volume (32,32,N_frames). In numpy: `pixels * mask[np.newaxis]`.
MATLAB usa dimensioni (32,32,N) — noi usiamo (N,32,32), ma il calcolo è equivalente.

**CreateLungContour.m** → Task 4.4 (lung mask):
Crea la maschera dai pixel con ventilazione sopra una soglia (configurabile 0-1).
Supporta erosione/dilatazione morfologica per pulire la maschera. In scipy:
`scipy.ndimage.binary_erosion / binary_dilation` con structuring element circolare.

**FindDeltas.m** → Task 5.3.1 (delta Z pixel-per-pixel):
Per ogni pixel calcola max, min, delta (max-min) in una finestra temporale.
In numpy vettorizzato: `np.max(pixels, axis=0) - np.min(pixels, axis=0)` su un volume
(N, 32, 32) tagliato a un singolo atto respiratorio.

### Orientamento pixel — confermato definitivamente

Da createROIMasks.m la convezione è inequivocabile:
- **Riga 0 = VENTRALE** (anteriore, parete toracica anteriore)
- **Riga 31 = DORSALE** (posteriore, schiena)
- **Colonne 0-15 = DESTRA** del paziente
- **Colonne 16-31 = SINISTRA** del paziente

Il nostro `BinData.roi_signals` è corretto. La ROI 0 (righe 0-7) è ventrale, ROI 3 (24-31) è dorsale.

### Note tecniche utili per dopo

- `np.frombuffer(buf, dtype=FRAME_BASE_DTYPE)` è il modo per parsare un frame
- `np.fromfile(path, dtype=FRAME_BASE_DTYPE)` legge tutto il file in una riga (Task 1.2.2)
- NaN sentinel in Medibus: `data[data < -1e30] = np.nan` (da eitprocessing)
- Per stimare fs dal file reale: `1 / scipy.stats.linregress(np.arange(n), time).slope`
  dove `time = frames["ts"] * 86400` con np.unwrap

### Stato tasks aggiornato
- Task 1.2.1: ✅ done — `src/fasteit/dtypes.py` con FRAME_BASE_DTYPE (4358b) e FRAME_EXT_DTYPE (4382b)
- Task 1.2.1b: ✅ done — BinData.timestamps aggiornato + min_max_flags + event_markers aggiunti
- Tests: `tests/test_dtypes.py` — 20 test verdi

### Analisi comparativa architettura: eitprocessing vs fastEIT

#### eitprocessing — flat container espandibile

Il loro contenitore centrale è `Sequence`. Non è un container grezzo: è già il risultato finale
che tiene tutto insieme dal primo momento in un unico oggetto con 4 DataCollection:
- `eit_data`: pixel impedance frames (EITData)
- `continuous_data`: waveform Medibus + segnali derivati (ContinuousData)
- `sparse_data`: eventi, marker respiro (SparseData)
- `interval_data`: intervalli temporali, respiri (IntervalData)

Quando carichi un file, ottieni subito una Sequence con tutto. Se poi applichi preprocessing,
**aggiungi altri oggetti alla stessa Sequence** con label diversi. Tutto vive in un unico posto.

Flessibilità orizzontale: puoi aggiungere qualsiasi segnale derivato con qualsiasi label.
Ottima per la ricerca esplorativa. Non ottimale per una pipeline clinica automatizzata.

#### fastEIT — pipeline a layer verticali

```
BinData    = sempre grezzo, sempre immutabile dopo parse
Session    = sempre il risultato di uno specifico preprocessing + Config
Breath     = sempre un singolo atto respiratorio con feature tipizzate
```

Se il breath detection è sbagliato → riparto da Session senza rileggere il file.
Se la lung mask è sbagliata → riparto da BinData senza riprocessare i respiri.
I layer sono separabili e riproducibili.

**La flessibilità è uguale ma in direzione diversa:**
- eitprocessing: flessibilità orizzontale (tutto nello stesso container)
- fastEIT: flessibilità verticale (ogni layer trasforma il precedente)

Per una pipeline su 200 pazienti la flessibilità verticale è quella giusta.

#### Cosa importiamo da eitprocessing (concetti, non codice GPL)

**1. `lock()` su numpy array → BinData.__post_init__**
Dopo il parse, BinData.frames non dovrebbe mai essere modificato.
eitprocessing usa `array.flags["WRITEABLE"] = False`.
Implementazione: aggiungere in BinData.__post_init__ dopo aver calcolato n_frames/duration.
Se qualcuno prova a scrivere `bin_data.frames["pixels"][0] = 0` → RuntimeError esplicito
invece di un bug silenzioso che modifica i dati grezzi.

**2. `select_by_time()` → Session (Task 4.13.x)**
Tutta la gerarchia eitprocessing supporta lo slicing temporale.
Per fastEIT essenziale per: estrarre finestre PEEP step, confrontare pre/post manovra,
debuggare sezioni specifiche di una registrazione.
```python
session.select_by_time(start_time=120.0, end_time=240.0)  # 2 minuti specifici
```

**3. `Config` come campo di Session**
Loro non hanno un modo esplicito per sapere con quali parametri è stata processata una Sequence.
Session dovrebbe salvare il Config usato:
```python
@dataclass
class Session:
    raw: BinData    # riferimento al dato grezzo
    config: Config  # config usato per il preprocessing
```
Quando pubblichi un paper, puoi riportare esattamente cosa è stato fatto.

**4. Validazione prima della concatenazione (Study.from_folder)**
Prima di `seq1 + seq2` eitprocessing verifica fs, vendor, categoria compatibili.
Noi dobbiamo fare lo stesso in Study.from_folder (Task 6.3.1): verificare che tutti
i BinData abbiano lo stesso fs prima di aggregarli.

#### Cosa NON vogliamo da eitprocessing
- DataCollection[V] con TypeVar generici — troppa macchina per il nostro scope
- Mancanza di confine raw/processed — problema per riproducibilità clinica
- Nessun Breath con feature tipizzate — è il nostro differenziatore principale
- Nessun PEEP step detection — il nostro differenziatore clinico
- Nessun export breath-level per database — il nostro differenziatore operativo

---

### Studio del file `slicing.py` di eitprocessing

File: `/home/ric/usefullfile/eitprocessing/slicing.py`
Copiato da `/home/ric/Downloads/slicing.py` nella sessione 2026-03-19.

#### Il pattern di design: ABC mixin a 3 livelli

```
SelectByIndex(ABC)          → aggiunge __getitem__ (obj[10:100]) + abstract _sliced_copy()
     ↓ eredita
SelectByTime                → aggiunge select_by_time(start, end) con bisect
     ↑ anche
HasTimeIndexer              → aggiunge .t property → TimeIndexer (obj.t[10.0:120.0])
```

`SelectByTime` eredita da entrambi → classe concreta eredita da `SelectByTime` e implementa
solo `_sliced_copy()`. Tutto il resto (slicing per indice, slicing per tempo, la `.t` property)
viene gratis dai mixin.

#### Due interfacce per fare la stessa cosa (pythonic)

```python
# Forma esplicita:
session.select_by_time(start_time=10.0, end_time=120.0)

# Forma sintetica (via TimeIndexer):
session.t[10.0:120.0]
```
`session.t` ritorna un oggetto `TimeIndexer` che sa come convertire una slice di float
in una chiamata a `select_by_time`. Molto elegante.

#### L'algoritmo chiave: `bisect`

`bisect` è un modulo Python standard che fa ricerca binaria su liste ordinate — O(log n).
Perfetto per trovare un timestamp in un array di 50×60×120 = 360.000 frame senza scorrere tutto.

Come funziona start_inclusive/end_inclusive:
```
bisect.bisect_left(time, t)   → indice del primo valore ≥ t  (includi t)
bisect.bisect_right(time, t)  → indice del primo valore > t  (escludi t se esiste uguale)
bisect.bisect_right(...) - 1  → ultimo indice ≤ t            (includi t, ritorna il frame "prima")
bisect.bisect_left(...) + 1   → primo indice > t             (includi t, ritorna il frame "dopo")
```

Per fastEIT la scelta default sensata è `start_inclusive=True, end_inclusive=False`,
così `session.t[10.0:120.0]` include il frame a t=10 e esclude il frame a t=120 —
uguale al comportamento di Python `list[10:120]`.

#### Come adatteremo per Session

```python
# Nella classe Session, basta implementare _sliced_copy:
def _sliced_copy(self, start_index: int, end_index: int, newlabel: str) -> Session:
    sliced_frames = self.raw.frames[start_index:end_index]
    sliced_raw = BinData(frames=sliced_frames, fs=self.raw.fs, ...)
    return Session(
        raw=sliced_raw,
        config=self.config,    # Config rimane invariato
        # tutti i segnali derivati sliced allo stesso modo
        lung_mask=self.lung_mask,       # 2D, non dipende dal frame — rimane invariata
        global_impedance=self.global_impedance[start_index:end_index],
        # ...
    )
```

La `lung_mask` è 2D (32×32) — non ha dimensione temporale, non si slice.
I segnali (global_impedance, roi_signals, Medibus) sono 1D → si slicano con start:end.

#### Note per il paper

Questo file è rilevante per la sezione **Methods** del paper:
- L'approccio `select_by_time` che implementeremo in Session verrà citato come
  "ispirato all'interfaccia di eitprocessing (apache-2.0)" con confronto esplicito
  della scelta architetturale (layered vs flat).
- La scelta `bisect` (O log n) sarà documentata nella sezione implementazione.
- Il confronto architetturale fastEIT vs eitprocessing sarà il cuore della sezione
  **Software Design** del paper: "while eitprocessing follows a flat expandable
  container pattern, fastEIT adopts a strict 3-layer pipeline to enforce reproducibility
  and enable structured clinical export."

---

### Informazione importante per Task 3.x: formato .asc

Il file CSV/TXT esportato dal PulmoVista ha estensione **.asc** (non .txt o .csv come pensavamo).
Va convertito con encoding **latin-1** (ISO-8859-1) prima del parsing.
CsvParser dovrà:
1. Accettare estensioni `.asc`, `.txt`, `.csv`
2. Aprire con `encoding="latin-1"` (o `encoding="iso-8859-1"`)
3. Autodetect del separatore (probabilmente `;` o `\t` come export PulmoVista)

Aggiornare Task 3.3.1 (CsvParser) con questa info quando ci arriveremo.

### Fine sessione 2026-03-19

**Task completate oggi:**
- Task 1.2.1: `dtypes.py` — FRAME_BASE_DTYPE (4358b) + FRAME_EXT_DTYPE (4382b) definitivi
- Task 1.2.1b: BinData aggiornata (timestamps, min_max_flags, event_markers)
- 50/50 test verdi, ruff pulito
- Studio architettura eitprocessing + analisi slicing.py
- Aggiornati: layerdata_class_architecture.md, reference_DEFINITIVO.md, learning_journal.md

**Prossima sessione — Task 1.1.1: Reverse engineering .bin, .eit, .asc reali**

Riccardo si procura i 3 file reali dal PulmoVista. Poi:

1. **Hexdump .bin** — `xxd file.bin | head -80`
   - Primi 8 byte come float64 LE = frazione di giorno (~0.5-0.9 = mattina/sera)
   - Offset 12 = inizio pixel, primo float32 plausibile (~ordine 10-100 AU)
   - Verificare `file_size % 4358 == 0` oppure `% 4382 == 0`
   - Salvare in `docs/reverse_eng/01_hexdump_bin_frame.txt`

2. **Hexdump .eit** — `xxd file.eit | head -80` + `strings file.eit | head -30`
   - Cercare magic string `---Draeger EIT-Software---` nei primi byte
   - Identificare fine header ASCII / inizio dati binari
   - Salvare in `docs/reverse_eng/02_hexdump_eit_header.txt`

3. **File .asc** — encoding latin-1, separatore da scoprire
   - `file file.asc` per info tipo
   - `head -5 file.asc` per vedere header e colonne
   - Salvare campione in `docs/reverse_eng/03_asc_sample.txt`

Ogni artefatto viene documentato in `docs/reverse_eng/` con: comando usato,
output raw, interpretazione byte-per-byte. Questi diventeranno la sezione
"File Format Specification" del paper.

---

## Sessione 2026-03-19 — Reverse engineering .bin con file reali

### Task lavorate
- Task 1.3.1: Investigare 208 byte Medibus nel frame base
- Task 1.3.2: Investigare 24 byte extra frame esteso (PressurePod)
- Task 1.4.3: Verificare timestamp crescenti e plausibili
- Task 1.4.4: Verificare pixel range ragionevole
- Task 1.4.5: Verificare Medibus valori clinici
- Prodotto: `docs/reverse_eng/bin_format.md` (documentazione pubblicabile)

### File analizzati
- `patient01.bin` 50.4 MB, 11500 frame, 230 secondi di registrazione
- `patient02.bin` 53.2 MB, 12150 frame, 243 secondi
- Stesso dispositivo: Dräger PulmoVista 500, ASPE-0048, V1.30
- `patient01.get` 11.8 MB — formato nuovo, da investigare

### Concetti spiegati

- **Prova matematica del frame size**: dimensione file ÷ numero frame (contato nell'ASC) = 4382.0 esatto su entrambi i pazienti. Questo è il modo più elegante per confermare una struttura binaria: non hai bisogno di documentazione, basta un divisione.

- **Timestamp come frazione di giornata**: il campo `ts` è un float64 che conta i secondi del giorno divisi per 86400. Quindi 18:10:54 = (18*3600 + 10*60 + 54) / 86400 = 0.7576. La conferma arriva dal confronto con l'header ASCII del file `.eit` corrispondente, che riporta "Time: 18:10:54.015" — coincidenza al secondo.

- **Due sentinel distinti nel .bin Dräger**:
  - `0xFF7FC99E` ≈ -3.40e+38: MEDIBUS fisicamente non connesso (i canali waveform, idx 0-5)
  - `0xC47A0000` = -1000.0: dato non ancora calcolato (inizio registrazione, campi breath-averaged, idx 6+)
  - Il codice usa `nan_value = -1e30` come soglia, non come valore esatto: -3.4e38 < -1e30, quindi cattura entrambi. Questo è il modo corretto di gestire i sentinel nei file binari: non cercare il valore esatto ma usare una soglia.

- **Hexdump annotato**: leggere un hexdump vuol dire sapere dove inizia ogni campo. L'offset si calcola sommando le dimensioni: ts(8) + dummy(4) + pixels(4096) = 4108 = 0x100C. Da lì vedi il `min_max_flag`. È come leggere una mappa del tesoro.

- **PressurePod validation con cross-check interno**: i campi idx 54, 55, 56 mostrano valori 14.0, 15.32, -1.28 mbar. Se questi sono Paw, Pes, Ptp: Paw - Pes = 14.0 - 15.32 = -1.32 ≈ -1.28. La differenza di 0.04 è probabilmente rounding float32 o una piccola correzione interna Dräger. Questo tipo di cross-check interno è fondamentale nel rev engineering: se Ptp = Paw - Pes, allora hai capito i campi giusti.

- **Il `.get` è un formato nuovo**: 1024 byte/frame = 256 × float32. Ipotesi: matrice completa 16×16 di transimpedanze (tutte le 256 coppie elettrodo), mentre il `.bin` usa solo le 208 del pattern adjacent-drive. Se confermato, sarebbe utile per integrare algoritmi di ricostruzione alternativi (pyEIT supporta matrici complete).

- **Il `.eit` contiene raw data, il `.bin` le immagini ricostruite**: il `.eit` (63 MB) è più grande del `.bin` (50 MB) non solo per l'header ASCII, ma perché contiene anche le 208 transimpedanze grezze per frame (i dati "prima" della ricostruzione). Il `.bin` è un sottoinsieme: solo le immagini 32×32 già ricostruite dal firmware.

### Domande di Riccardo
- "I file grezzi .eit non li abbiamo dentro il file .eit??" → Risposta: sì! Il .eit contiene sia l'header ASCII (metadati), sia i dati grezzi (208 transimpedanze per frame) SIA probabilmente i segnali Medibus. È per questo che è più grande del .bin. Lo analizzeremo in dettaglio quando arriveremo al parser .eit (Task 2.x).

### Epifanie e insight
- "La prova più elegante che FRAME_EXT_DTYPE è giusto è una sola divisione: 50393000 ÷ 11500 = 4382.0 esatto. Se ci fosse anche un solo byte sbagliato, non darebbe un numero intero."
- "Il timestamp ts[0] = 18:10:54 che leggo dal file .bin coincide ESATTAMENTE con il campo Time nell'header ASCII del .eit. Due file, stesso secondo. È la prova che .bin e .eit sono paralleli e sincronizzati."
- "I sentinel -3.4e38 e -1000.0 sono in realtà due informazioni distinte: -3.4e38 dice 'il sensore non c'è', -1000.0 dice 'il sensore c'è ma non ha ancora un valore'. Come NULL vs 0 in un database."
- "Il PressurePod validation cross-check: misuro 14.0, 15.32, -1.28 e quando faccio 14.0-15.32 ottengo -1.32 ≈ -1.28. In quel momento so con certezza di aver identificato Paw, Pes e Ptp — senza documentazione Dräger."

### Note tecniche utili per dopo
- `np.memmap` invece di `np.fromfile` permette accesso lazy senza caricare tutto in RAM — utile per file da 50+ MB
- Il naming dei campi PressurePod (idx 52-57) è probabilmente sfasato: il campo chiamato `time_at_low_pressure_pod` (idx 54) in realtà contiene Paw (10-38 mbar). Da chiarire con il parser .eit.
- Per verificare il `.get` su patient02: `wc -c patient02.get` → se = 12,441,600 byte (= 12150 × 1024), la struttura 256×float32 è confermata.
- `xxd -l N file.bin` per vedere i primi N byte; annotare gli offset calcolati manualmente prima di leggere l'output


---

## Sessione 2026-03-19 (continua) -- eit_format.ipynb

### Task lavorate
- Task 2.1.2-2.1.5: Hexdump, ASCII header, magic string, metadati .eit
- Creato: `docs/reverse_eng/eit_format.ipynb` (22 celle, 8 code con output pre-saved)
- Aggiunto Task 2.1.6: struttura interna frame .eit

### Concetti spiegati

- **Struttura header .eit**: il file inizia con 12 byte di preamble binario (3 x int32 LE), poi un blocco ASCII leggibile con tutti i metadati device/paziente, poi un separatore `**\r\n\r\n\r\n` (8 byte), poi il blocco binario. Il valore `preamble[1]` (es. 7546) punta esattamente all'inizio del separatore: `binary_start = preamble[1] + 8`.

- **Dimensione frame .eit**: 5495 byte/frame (vs 4382 byte/frame del .bin). Verificato su entrambi i pazienti: p01 = (63200054 - 7554) / 5495 = 11500.0 esatto; p02 = (66771796 - 7546) / 5495 = 12150.0 esatto.

- **Header varia tra pazienti**: `preamble[1]` = 7546 per p01, 7538 per p02. L'header ASCII ha lunghezza variabile (dipende da lunghezza dei campi testo come nome paziente). Il parser .eit DEVE usare `preamble[1]` per trovare il binary_start, non un offset fisso.

- **Frame .eit: 5487 byte dopo il timestamp**: 5487 è dispari (5487 % 4 = 3), quindi c'è almeno un campo non-float32. La struttura interna è ancora da determinare. Ipotesi: parte measurements (208 complex float32 = 1664 byte) + parte pixel ricostruiti (1024 float32 = 4096 byte) ma 8+1664+4096 = 5768 ≠ 5495. Mancano ~273 byte di discrepanza -- da risolvere con EIDORS analysis (Task 2.0).

- **Sentinel nel .eit**: il valore -3.4e12 appare anche nel .eit, probabilmente lo stesso sentinel `0xFF7FC99E` del .bin per elettrodi disconnessi.

### Domande di Riccardo
- (nessuna domanda esplicita questa sotto-sessione -- lavoro tecnico autonomo)

### Epifanie e insight
- "preamble[1] varia tra pazienti perché l'header ASCII ha lunghezza variabile! Quindi il parser NON può assumere binary_start fisso ma deve leggere i 4 byte a offset 4 e aggiungere 8."
- "5495 byte/frame è un numero strano che non si fattorizza bene -- il frame .eit non è progettato per essere 'pulito' come il .bin (4382 = 4096+286). Probabilmente ci sono campi legacy o padding."
- "La prova frame size sul .eit è identica a quella del .bin: una divisione che deve dare un intero esatto. Se il file è corrotto o il frame_size sbagliato, il resto non è zero."

### Note tecniche utili per dopo
- **binary_start formula**: `binary_start = struct.unpack('<i', data[4:8])[0] + 8` (sempre, per qualsiasi file formato 51)
- **Frame size .eit**: 5495 byte, confermato su 2 pazienti diversi. Non divisibile per 4.
- **Medibus nel .eit?**: ancora da determinare se i canali Medibus sono anche nel .eit o solo nel .bin. Dal .asc (export software) sono presenti waveform channels -- potrebbero venire dal .eit o essere aggiunti dal software.
- Per Task 2.0 (EIDORS): cercare `fread` con 5495 o 5487 nel codice EIDORS per trovare la definizione della struttura frame

### Sessione 2026-03-19 (continua) -- Debug notebook `bin_format.ipynb`

#### Task lavorate
- Debug di `docs/reverse_eng/bin_format.ipynb`: le prime celle non partivano da kernel pulito
- Fix delle celle di setup path e hexdump per renderle indipendenti dalla working directory

#### Concetti spiegati

- **Working directory di un notebook**: una path relativa come `src/fasteit/test_files/patient01.bin` funziona solo se il kernel parte dalla root del repo. Se il notebook viene eseguito da `docs/reverse_eng/`, quella stessa stringa punta a `docs/reverse_eng/src/...` e fallisce anche se il file esiste davvero.

- **Path robusti in notebook**: invece di assumere dove si trova il processo, conviene cercare la root del progetto risalendo le parent directory fino a trovare marker stabili (`pyproject.toml` + `src/fasteit`). Da lì costruisci `TEST_FILES = REPO_ROOT / 'src' / 'fasteit' / 'test_files'` e tutto il notebook resta ri-eseguibile.

- **Kernel sporco vs notebook lineare**: il fatto che alcune celle successive girassero non significava che il notebook fosse corretto. Significava che il kernel aveva già variabili o stato residuo. La prova vera è: restart kernel, run all from top.

#### Epifanie e insight
- "Il problema non era il contenuto del `.bin`, ma il fatto che il notebook dava per scontato da dove veniva lanciato. In un notebook i path relativi sono fragili quanto gli import fatti a mano."
- "Se una cella fallisce ma quelle dopo vanno, non è una prova che il notebook funzioni: è spesso il segnale opposto, cioè che dipende da stato nascosto del kernel."

#### Note tecniche utili per dopo
- Fix applicato: prima cella ora definisce `REPO_ROOT` e `TEST_FILES` con `pathlib.Path`
- Fix applicato: cella hexdump usa `np.memmap(TEST_FILES / 'patient01.bin', ...)` invece di stringhe relative hardcoded
- Verifica eseguita: restart kernel + run top-to-bottom di tutte le celle di codice -> tutte eseguite con successo

---

## Sessione 2026-03-19 (continua) -- PressurePod Pgas identificata

### Concetti spiegati
- **PressurePod Dräger**: accessorio con esattamente 3 connessioni fisiche -- Paw (pressione vie aeree, tap sul circuito ventilatorio), Pes (pressione esofagea, catetere a palloncino), Pgas (pressione gastrica, catetere a palloncino gastrico). Deriva Ptp = Paw - Pes internamente.
- Questo definisce completamente i 4 campi waveform (idx 54-57) senza ambiguità.

### Epifanie e insight
- "Se quello che ci manca è solo un index, allora sicuramente è la pressione gastrica perché il PressurePod registra esofagea, airway pressure così calcola la Ptp e la gastrica -- ha questi tre attacchi e basta."
- Ragionamento clinico diretto: 3 input fisici + 1 derivata = 4 campi waveform. Non c'è posto per altro. Questo tipo di ragionamento (conoscenza dell'hardware) risolve ambiguità che nessun hexdump da solo può risolvere.
- **Bug scoperto**: i nomi in `_MEDIBUS_EXT_EXTRA` in `dtypes.py` sono sfasati di 1 posizione. `time_at_low_pressure_pod` (idx 54) = Paw (14 mbar, non secondi). Il fix è rinominare: idx 54=`airway_pressure_pod`, 55=`esophageal_pressure_pod`, 56=`transpulmonary_pressure_pod`, 57=`gastric_pressure_pod`.

### Mapping completo PressurePod (idx 52-57)
| idx | nome attuale (sbagliato) | nome corretto | tipo |
|-----|--------------------------|---------------|------|
| 52 | `high_pressure` | `high_pressure` (ok) | per-breath peak Paw |
| 53 | `low_pressure` | `low_pressure` (ok) | per-breath PEEP |
| 54 | `time_at_low_pressure_pod` | `airway_pressure_pod` | waveform continuo |
| 55 | `airway_pressure_pod` | `esophageal_pressure_pod` | waveform continuo |
| 56 | `esophageal_pressure_pod` | `transpulmonary_pressure_pod` | waveform continuo |
| 57 | `transpulmonary_pressure_pod` | `gastric_pressure_pod` | waveform continuo |

### Note tecniche utili per dopo
- Il fix in `dtypes.py` è SOLO rinominare le ultime 4 tuple di `_MEDIBUS_EXT_EXTRA` -- non cambia la struttura dati
- Valore Pgas = 0.04 mbar è fisiologicamente plausibile: pressione gastrica a riposo vicina allo zero in assenza di contrazione diaframmatica significativa

---

## Sessione 2026-03-19 (continua) -- Fix dtypes.py + Analisi prototipo + Fix BiLevel fields

### Epifanie e insight (BiLevel fields)
- "Il CSV del prototipo era la Rosetta Stone che ci mancava per risolvere definitivamente i nomi dei campi Medibus"
- Confrontando le colonne del CSV (che il PulmoVista genera in parallelo al .bin) con la posizione numerica: col 63 = medibus idx 51 = `phigh_mbar`. Non `time_at_low_pressure` come pensavamo.
- **BiLevel/APRV**: è una modalità ventilatoria a due livelli di pressione (Phigh = pressione alta, Plow = pressione bassa, Tlow = tempo a pressione bassa). Il PulmoVista registra questi tre parametri in Medibus.
- **Phigh è nel formato BASE** (idx 51, ultimo campo delle 52 floats): significa che tutti i frame base (4358 byte) hanno già Phigh, indipendentemente dal PressurePod. Per pazienti su ventilazione convenzionale questo campo sarà sentinel o zero.
- Plow e Tlow sono EXT perché... probabilmente aggiunti in un firmware successivo? O perché meno usati? Non importa -- il CSV ce lo dice chiaramente.

### Correzioni finali dtypes.py (tutti e 3 i fix applicati nella stessa sessione)
| idx | SBAGLIATO | CORRETTO | Dove |
|-----|-----------|---------|------|
| 51 | `time_at_low_pressure` (unità "s") | `high_pressure` (unità "mbar") | MEDIBUS_FIELDS base |
| 52 | `high_pressure` ("mbar") | `low_pressure` ("mbar") | _MEDIBUS_EXT_EXTRA |
| 53 | `low_pressure` ("mbar") | `time_at_low_pressure` ("s") | _MEDIBUS_EXT_EXTRA |
| 54 | `time_at_low_pressure_pod` ("s") | `airway_pressure_pod` ("mbar") | (fix precedente) |
| 55 | `airway_pressure_pod` | `esophageal_pressure_pod` | (fix precedente) |
| 56 | `esophageal_pressure_pod` | `transpulmonary_pressure_pod` | (fix precedente) |
| 57 | `transpulmonary_pressure_pod` | `gastric_pressure_pod` | (fix precedente) |

---

## Sessione 2026-03-19 (continua) -- Fix dtypes.py + Analisi prototipo

### Task completate
- **Task 1.3.5**: Fix `_MEDIBUS_EXT_EXTRA` in `dtypes.py` — rinominati idx 54-57, rimosso `time_at_low_pressure_pod`, aggiunto `gastric_pressure_pod`. Test aggiornati: 22/22 verdi.
- **Task 1.3.6**: Analisi prototipo `mia_vecchia_prova/` — documentata in `docs/prototype_analysis.md`
- **Task 1.4.6 / 2.1.7**: Aggiunte al CSV (già completate in sessione precedente con i notebook)

### Concetti appresi dal prototipo
- **Breath detection con derivata centrale**: `d[i] = (y[i+1] - y[i-1]) / 2 * fs` — trova l'inizio inspirazione quando la derivata passa da negativa a positiva (la curva di impedenza globale risale). Semplice ma funzionante come prova di concetto. Il miglioramento in Task 4.10.x userà `find_peaks` con parametri adattivi.
- **PEEP detection con threshold crossing**: paragona delta frame-to-frame di Paw smoothata con una soglia (0.9 mbar). Se il delta supera la soglia → nuovo livello. Intuitivo ma fragile: un rumore transitorio può far scattare un nuovo livello. In Task 5.7.x si userà analisi dei segmenti stabili.
- **BreathFeatures**: 11 statistiche (mean/std/var/skew/kurtosis/min/max/median/p25/p75/IQR) per 8 segnali = 88 features statistiche + 3 temporali (slope/acceleration/smoothness) + 3 inter-breath (delta_peak/delta_tidal/delta_duration) + ~18 features base = ~111 features per respiro. Questa è la "golden list" di spec per Task 4.x–5.x.
- **Encoding latin-1**: i file .asc del PulmoVista non sono UTF-8. La virgola come separatore decimale è standard europeo. Da gestire in Task 3.x.

### Epifanie e insight
- "Il prototipo è esattamente quello che fa il PulmoVista ma in Python — e lui funzionava su file reali! Quindi la logica è validata clinicamente anche se il codice è grezzo."
- Il `to_dict()` e `to_features_dict()` del prototipo sono di fatto la **specifica di interfaccia** della futura classe `Breath` — invece di reinventare da zero, si formalizza quello che già funziona.
- `pgapaux3_pod_mbar` nel prototipo = `gastric_pressure_pod` (idx 57) — il prototipo leggeva questo campo dall'ASC già correttamente, confermando il nostro reverse engineering.

### Note tecniche utili per dopo
- Il prototipo usa `df["insp_t0"] == True` come condizione di split — in pandas moderno è meglio `df["insp_t0"]` direttamente (evita warning FutureWarning)
- `BreathFeatures` eredita da `Breath` (inheritance) — in fastEIT la struttura sarà diversa (composition o dataclass), ma le formule sono le stesse
- Il mapping `standardize_labels()` è il rosetta stone per verificare che i nomi in `MEDIBUS_FIELDS` corrispondano ai nomi dell'ASC (vedere `docs/prototype_analysis.md` sezione mapping)

---

## Sessione 2026-03-19 (continua) — REVERT fix errato idx 51/52/53

### Errore commesso e corretto

**Cosa è successo**: nella sessione precedente abbiamo fatto un fix SBAGLIATO basato
sull'assunzione che "ordine colonne CSV ASC = indici binari medibus_data".

Avevamo cambiato:
- base idx 51: `time_at_low_pressure` → `high_pressure`
- ext idx 52: `high_pressure` → `low_pressure`
- ext idx 53: `low_pressure` → `time_at_low_pressure`

**Perché era sbagliato — la prova**:
- `phigh_mbar`, `plow_mbar`, `tlow_s` nel CSV dei 2 pazienti analizzati sono **tutti NaN/zero**
  (i pazienti NON erano in modalità BiLevel/APRV)
- I campi binary idx 52 e 53 mostravano invece **valori non-zero** (pressioni per-breath)
- Conclusione: binary idx 52/53 ≠ BiLevel params. Sono valori PressurePod-derived per-breath
  (peak Paw e PEEP-equiv calcolati dal PressurePod sul singolo respiro)
- L'ordine colonne nell'ASC **NON corrisponde** agli indici nel binary medibus_data — sono
  sistemi separati con ordinamenti diversi

**Stato CORRETTO ripristinato**:
| idx | nome | unità | tipo | prova |
|-----|------|-------|------|-------|
| 51 | `time_at_low_pressure` | s | base | zero nei non-BiLevel — atteso e corretto |
| 52 | `high_pressure` | mbar | ext | non-zero nei file reali — peak Paw per-breath |
| 53 | `low_pressure` | mbar | ext | non-zero nei file reali — PEEP-equiv per-breath |

### Lezione appresa

**Regola**: il CSV ASC e il binario `.bin` sono due esportazioni **parallele e indipendenti**
dello stesso dispositivo. Hanno nomi di colonna e struttura propri. Non si può assumere che
la N-esima colonna del CSV corrisponda all'N-esimo campo del binary.

**Come evitarlo in futuro**: per validare un indice binary, servono pazienti con il campo
attivo (BiLevel per idx 51, PressurePod connesso per idx 52/53). Con pazienti in ventilazione
convenzionale i campi BiLevel sono sempre zero — non si può distinguere "zero perché non in
BiLevel" da "zero perché sbagliamo l'indice".

### Task da aggiungere al CSV

**Task 1.4.7**: "Procura e analizza recording .bin/.asc senza PressurePod" — prova del nove
definitiva per:
1. Confermare frame size base (4358 byte)
2. Verificare idx 51 con un paziente BiLevel (se disponibile)
3. Vedere quali colonne ASC appaiono senza PressurePod

### Files modificati in questo revert
- `src/fasteit/dtypes.py` — idx 51 e _MEDIBUS_EXT_EXTRA idx 52/53
- `tests/test_dtypes.py` — `test_medibus_index_lookup` (3 assertions corrette)
- `docs/prototype_analysis.md` — tabella mapping correta + nota BiLevel

---

## Sessione 2026-03-27 — GREIT reconstruction + validazione segnale globale

### Task lavorate
- Task 7.1.1–7.1.3: integrazione pyEIT, notebook GREIT
- Task 7.2.1: `eit_greit.py` — `build_greit()` + `reconstruct_greit()`

### Concetti spiegati

- **GREIT (Adler 2009)**: ricostruzione EIT differenziale. Produce Δσ rispetto a un frame di riferimento (v0). Output flat (n²,) → reshape (32,32). Parametri default: p=0.2, lamb=1e-2, n=32. pyEIT: `mesh.create` + `protocol.create` + `GREIT.setup` + `solve(v1, v0)`.

- **Convenzione segno**: GREIT produce Δσ (conduttività) → decresce quando l'aria entra nei polmoni. La convenzione clinica (e il .bin Dräger) usa l'impedenza → cresce con l'aria. **Fix: negare l'output**. Scoperto empiricamente confrontando i plot: le "punte" GREIT cadevano tra le punte del .bin invece di sovrapporsi.

- **Orientamento anatomico**: pyEIT mette l'elettrodo 1 a ore 3 (destra del cerchio). Il PulmoVista ha l'elettrodo 1 anteriore. Confrontando visivamente le 4 rotazioni possibili (0°, 90°CCW, 180°, 90°CW), la 90°CCW posiziona correttamente l'area cardiaca a sinistra-anteriore. **Fix: `np.rot90(image, k=1)` nell'output di `reconstruct_greit()`**.

- **ref_frame e EELI cross-file**: GREIT è differenziale. Con `ref_frame=None` (default = media di tutto il file) ogni file è centrato sulla sua media → EELI ≈ 0 per costruzione → non confrontabile tra file. Per confrontare EELI tra registrazioni (es. NIV interfaccia A vs B) servono tutte sullo stesso riferimento esterno: `ref = baseline[:50].mean(axis=0)` applicato a tutti i file. **Limite**: se il device ricalibrasi tra i file, l'offset non è recuperabile.

- **Ricalibrazione PulmoVista**: il device ricalibrasi automaticamente, resettando il suo zero interno. Per questo l'EELI nel software Dräger può partire da 0 o da 1.000.000 — sono valori assoluti arbitrari che dipendono dall'ultimo reset. Non confrontabili tra sessioni senza un riferimento fisiologico fisso. Il software Dräger non lo esplicita, fastEIT sì.

### Epifanie e insight

- **"GREIT senza filtro ≈ .bin raw"**: confrontando `global_greit` (nansum dei pixel) con `bin_.global_signal`, le curve si sovrappongono dopo negazione senza alcun filtro. Significa che o il Dräger non filtra internamente (ipotesi più probabile, come confermato dal fatto che eitprocessing e colleghi MATLAB devono filtrare da soli), oppure la regularizzazione GREIT agisce da sola come filtro spaziale.

- **"Le punte erano al contrario"**: scoperta della convenzione segno semplicemente guardando che i picchi GREIT cadevano negli "spazi" tra i picchi .bin. Intuizione visiva immediata senza calcoli.

- **"L'area rossa fissa a sinistra = cuore"**: osservando che una zona ad alta impedenza rimaneva fissa (non respiratoria) nella parte sinistra dell'immagine, Riccardo ha ipotizzato la presenza del cuore. Questo ha portato a scoprire il problema di rotazione degli elettrodi.

### Note tecniche utili per dopo
- `np.nansum(images, axis=(1,2))` → segnale globale EIT (confrontabile con `bin_.global_signal`)
- Negazione + 90°CCW sono già built-in in `reconstruct_greit()` — non applicare manualmente
- Per EELI cross-file: sempre passare `ref_frame` esterno dalla registrazione baseline della stessa sessione
- `n_meas = protocol.n_exc * protocol.n_meas = 16 * 13 = 208` ← corrispondenza esatta col formato .eit Dräger
- Nuovo agente: `eit-physics-reviewer` — da chiamare quando si implementano formule EIT o features cliniche
