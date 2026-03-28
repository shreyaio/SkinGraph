# SkinGraph — Phase-Wise Implementation Plan
## Agent-Executable, Chunk-by-Chunk Build Order
### Source of Truth: Architecture-Workflow-Plan.md

---

## HOW TO READ THIS DOCUMENT

Every section is a self-contained execution unit. Each has:
- **What it does** — the purpose in plain terms
- **Files to create** — exact paths
- **What each file does** — described theoretically, no code
- **Cross-file updates** — any previously created file that needs to be touched
- **Exit condition** — how you know this section is done before moving to the next

Do not start a section until the previous section's exit condition is met. Sections within the same phase can sometimes run in parallel — this is marked explicitly. If not marked, treat as sequential.

---

# PHASE 0 — SKELETON AND TYPES
## "Nothing runs yet, but everything knows its shape"

---

### Section 0.1 — Project Scaffold [DONE]

**What it does:** Creates the entire directory tree and every `__init__.py`. No logic, no imports, just structure. After this section, the project is navigable and importable.

**Files to create:**
- `skingraph/` — root project folder
- `skingraph/data/raw/pubmed_abstracts/` — empty dir
- `skingraph/data/raw/ewg_scraped/` — empty dir
- `skingraph/data/processed/` — empty dir
- `skingraph/data/graph/` — empty dir
- `skingraph/ingestion/__init__.py` — empty
- `skingraph/nlp/__init__.py` — empty
- `skingraph/graph/__init__.py` — empty
- `skingraph/scoring/__init__.py` — empty
- `skingraph/models/__init__.py` — empty
- `skingraph/models/artifacts/` — empty dir (will hold .pkl)
- `skingraph/api/__init__.py` — empty
- `skingraph/api/routes/__init__.py` — empty
- `skingraph/explainer/__init__.py` — empty
- `skingraph/tests/__init__.py` — empty
- `skingraph/tests/fixtures/` — empty dir
- `skingraph/scripts/` — empty dir
- `skingraph/config/__init__.py` — empty
- `.gitignore` — ignore `.env`, `data/raw/`, `data/graph/`, `models/artifacts/`, `__pycache__/`, `.DS_Store`
- `.env.example` — template showing all required env vars: `GROQ_API_KEY`, `ENTREZ_EMAIL`, `PUBCHEM_API_KEY`

**Exit condition:** Running `find skingraph -type d` shows all expected directories. Every `__init__.py` exists.

---

### Section 0.2 — Requirements Files [DONE]

**What it does:** Pins every dependency the project needs. Split into two files: one for the main machine (CPU inference + API), one for Colab (GPU training). This prevents version conflicts and documents intent.

**Files to create:**

`requirements.txt` — contains:
- `fastapi` and `uvicorn[standard]` — API server
- `pydantic` and `pydantic-settings` — request/response validation and settings
- `networkx` — knowledge graph
- `spacy` — NLP tokenisation and dependency parsing
- `biopython` — Entrez/PubMed API wrapper (this is the standard NCBI Python client)
- `requests` and `beautifulsoup4` — HTTP fetching and HTML scraping
- `rapidfuzz` — fuzzy string matching for ingredient normalisation
- `pandas` — CSV manipulation throughout the pipeline
- `joblib` — model serialisation/deserialisation
- `lightgbm` — ML model inference (CPU only, no GPU flag needed for inference)
- `groq` — Groq API client
- `python-dotenv` — loads `.env` file
- `pytest` and `httpx` — testing

`requirements_colab.txt` — contains everything above plus:
- `lightgbm` with GPU build instructions (comment noting Colab install command)
- `scikit-learn` — for train/test split and AUC-ROC evaluation metrics
- `matplotlib` and `seaborn` — for feature importance plots during training

**Exit condition:** Both files exist and `pip install -r requirements.txt` completes without errors on local machine.

---

### Section 0.3 — Configuration Module [DONE]

**What it does:** Creates the single configuration object that every other module will import. All paths, thresholds, weights, and API keys live here. No module hardcodes any value — they all read from this object.

**Files to create:**

`config/settings.py`
- Defines a `Settings` class using `pydantic-settings BaseSettings`
- Reads from `.env` file automatically
- Fields: all file paths (graph, model, CSV), all API keys (Groq), all scoring thresholds (high/medium conflict cutoffs), all ensemble weights (the 0.5/0.3/0.2 split from architecture doc), all rate limit sleep intervals
- Instantiates a `settings` singleton at module level — every importer gets the same object
- No logic, only configuration values

`config/logging_config.py`
- Defines a `setup_logging()` function
- Configures Python's standard `logging` module with a structured format: timestamp, level, module name, message
- Sets root log level to INFO, suppresses noisy third-party logs (urllib3, etc.) to WARNING
- Returns a logger that any module can use by calling `logging.getLogger(__name__)`

**Exit condition:** `from config.settings import settings` works without error. `settings.GRAPH_PATH` returns the correct default string.

---

### Section 0.4 — Graph Schema (Type Definitions) [DONE]

**What it does:** Defines the shape of every data object in the system as Python dataclasses. This is the contract between all modules. The graph builder writes these shapes. The query functions return these shapes. The scoring engine reads these shapes. Nothing is a raw dict after this point.

**Files to create:**

`graph/schema.py`
- `NodeData` dataclass — all fields from the node schema in architecture doc: `canonical_inci`, `pubchem_cid`, `function_labels` (list), `irritancy_level`, `acne_risk`, `ewg_score`, `ph_optimal` (tuple), `skin_type_caution` (list), `fitzpatrick_notes`. All optional fields default to None or empty list.
- `EdgeData` dataclass — all fields from edge schema in architecture doc: `relation_type`, `base_score`, `source_count`, `agreement_ratio`, `primary_mechanism`, `fitzpatrick_iv_v_flag`, `pih_risk`, `confidence_tier`, `sources` (list), `safe_at_concentrations` (optional), `order_note` (optional)
- `RelationType` string enum — values: `CONFLICT`, `SYNERGY`, `ORDER_SENSITIVE`, `NEUTRAL`, `UNCLEAR`
- `SkinProfile` dataclass — `skin_type`, `fitzpatrick`, `sensitivity`, `concerns` (list), `allergies` (list), `history` (list)
- `ConfidenceReport` dataclass — `score` (float), `tier` (str), `source_count` (int), `uncertainty_note` (optional str)
- `RiskMatrix` dataclass — a wrapper around a list of `EdgeData` objects for a given ingredient set, plus a `cumulative_burden_score` float

**Exit condition:** `from graph.schema import NodeData, EdgeData, SkinProfile` works. Creating an instance of each with dummy data works without error.

---

### Section 0.5 — API Schemas (Request/Response Types) [DONE]
Added BaseSchema with arbitrary_types_allowed=True to ensure compatibility between Pydantic models and internal dataclasses (SkinProfile, ConfidenceReport). This prevents serialization and validation issues in FastAPI.

**What it does:** Defines all Pydantic models for the FastAPI layer. These are separate from the internal dataclasses because they handle JSON serialisation, validation error messages, and API documentation. They import from `graph/schema.py` where the types overlap.

**Files to create:**

`api/schemas.py`
- `ProductInput` model — `name` (str, required), `ingredients` (optional list of str), `routine_step` (optional, literal "AM"/"PM"/"both")
- `ScanRequest` model — `products` (list of `ProductInput`, min 2 max 5), `skin_profile` (optional `SkinProfile`)
- `ConflictResult` model — all fields from architecture doc: both ingredient names, both product names, `base_score`, `skin_adjusted_score`, `ml_score`, `final_score`, `severity`, `mechanism`, `confidence` (embedded `ConfidenceReport`), `recommendation`
- `SynergyResult` model — ingredient pair, product pair, synergy score, mechanism, recommendation
- `OrderResult` model — ingredient pair, product pair, correct order, what goes wrong if reversed
- `RoutineOutput` model — `am_routine` (ordered list of ingredient names), `pm_routine` (ordered list), `notes` (list of strings)
- `ScanResponse` model — `scan_id` (UUID str), `conflicts` (list), `synergies` (list), `order_issues` (list), `routine_recommendation`, `overall_risk_level`, `cumulative_burden_score`
- `ExplainRequest` model — wraps a `ScanResponse` plus `SkinProfile`
- `ExplainResponse` model — `explanation` (str), `word_count` (int)

**Exit condition:** `from api.schemas import ScanRequest, ScanResponse` works. Instantiating `ScanRequest` with two products validates correctly.

---

# PHASE 1 — DATA INGESTION
## "Fill the raw data folders. No graph yet, no scoring yet."

---

### Section 1.1 — Open Beauty Facts Fetcher

**What it does:** Downloads the Open Beauty Facts product database as a JSON dump and saves it to `data/raw/`. Extracts all unique ingredient names from all products and writes them to a flat list. This becomes the seed vocabulary for the master ingredient list.

**Files to create:**

`ingestion/openbf_fetcher.py`
- Function `download_dump()` — downloads the Open Beauty Facts products JSON dump from their public URL (`https://world.openbeautyfacts.org/data/en.openbeautyfacts.org.products.json.gz`), saves compressed file to `data/raw/openbf_products.json.gz`
- Function `extract_ingredient_names(dump_path)` — iterates through all products in the dump, extracts the `ingredients_text` field, splits on commas and common delimiters, applies basic cleaning (strip whitespace, lowercase, remove percentages), returns a deduplicated flat list of raw ingredient name strings
- Function `run()` — orchestrates download then extraction, writes raw ingredient list to `data/raw/openbf_ingredients_raw.txt` (one name per line), logs count of products processed and unique ingredients found

The fetcher checks if the dump already exists before downloading. If it exists, skip download and go straight to extraction. This is the cache-first pattern used by all fetchers.

**Exit condition:** `data/raw/openbf_ingredients_raw.txt` exists with 10,000+ unique ingredient strings.

---

### Section 1.2 — Ingredient Normaliser (Master List Builder)

**What it does:** This is the most important ingestion component. Takes the raw ingredient name list from Open Beauty Facts, deduplicates aggressively, and produces `ingredients_master.csv` — the canonical ingredient vocabulary the entire system uses. At this stage, PubChem lookups are NOT run (that happens in graph build). This file is just the name normalisation layer.

**Files to create:**

`nlp/normaliser.py`
- `IngredientNormaliser` class
  - Constructor takes path to `ingredients_master.csv` and loads it into memory as a dict mapping every alias → canonical INCI name
  - `normalise(raw_name: str) -> str` — the main function. Steps: (1) lowercase + strip, (2) exact match against alias dict, (3) if no match, run rapidfuzz against all aliases at threshold 85, return best match's canonical name, (4) if still no match, return the cleaned raw name marked as `UNKNOWN::<cleaned_name>` and log it
  - `normalise_list(names: List[str]) -> List[str]` — maps normalise over a list
  - `is_known(name: str) -> bool` — returns True if the name resolves to a known canonical ingredient
- Module-level function `build_master_csv(raw_names_path, output_path)` — takes the raw Open Beauty Facts ingredient list, applies basic grouping (same stem = same ingredient, e.g. "retinol", "retinols", "retinol usp"), writes `ingredients_master.csv` with columns: `canonical_inci_name`, `aliases` (pipe-separated), `source`. For V1, PubChem CID and EWG scores are empty — those are filled during graph build.

**Cross-file updates:** None yet, but note that this class will be imported by `api/main.py` (startup), `ingestion/pubmed_fetcher.py`, and `scripts/test_scan.py`.

**Exit condition:** `data/processed/ingredients_master.csv` exists. `from nlp.normaliser import IngredientNormaliser` works. `normaliser.normalise("vitamin a")` returns `"retinol"`.

---

### Section 1.3 — PubMed Fetcher

**What it does:** Queries the NCBI Entrez API for scientific abstracts about ingredient interactions. For each priority ingredient pair (a hardcoded seed list of ~50 pairs to start), searches PubMed and downloads abstracts. Saves raw XML responses to `data/raw/pubmed_abstracts/`.

**Files to create:**

`ingestion/pubmed_fetcher.py`
- Constant `PRIORITY_PAIRS` — hardcoded list of the 50 most important ingredient pairs to query. Includes all known conflict pairs (retinol + AHA, retinol + Vitamin C, BHA + Vitamin C, etc.) and known synergy pairs (niacinamide + zinc, hyaluronic acid + niacinamide, etc.)
- Function `build_query(ingredient_a, ingredient_b)` — constructs a PubMed search string like `"retinol"[tiab] AND "lactic acid"[tiab] AND ("skin" OR "topical" OR "dermal")`
- Function `fetch_abstracts_for_pair(ingredient_a, ingredient_b, max_results=50)` — uses Biopython's `Entrez.esearch` to get PMIDs, then `Entrez.efetch` to get full abstract XML. Sleeps `1/rate_limit` seconds between calls. Saves result to `data/raw/pubmed_abstracts/{ingredient_a}__{ingredient_b}.xml`. Skips if file already exists.
- Function `run_all_pairs()` — iterates `PRIORITY_PAIRS`, calls `fetch_abstracts_for_pair` for each, logs progress

**Cross-file updates:**
- `config/settings.py` — the `ENTREZ_EMAIL` field is used here. Import `settings` to get it.

**Exit condition:** `data/raw/pubmed_abstracts/` contains ~50 XML files, one per ingredient pair.

---

### Section 1.4 — PubMed NLP Extractor

**What it does:** Reads each XML abstract file, runs NLP to extract relationship triples, and writes them to `conflict_pairs_raw.csv` and `synergy_pairs_raw.csv`. This is the bridge between raw literature and graph edges.

**Files to create:**

`nlp/relation_types.py`
- `RelationType` string enum — `CONFLICT`, `SYNERGY`, `ORDER_SENSITIVE`, `NEUTRAL`, `UNCLEAR`
- `CONFLICT_VERBS` — curated list of verbs/phrases that indicate conflict: "irritates", "degrades", "deactivates", "conflicts", "inhibits", "counteracts", "destabilises", "overexfoliates", "increases sensitivity to"
- `SYNERGY_VERBS` — curated list indicating synergy: "enhances", "potentiates", "complements", "synergises", "boosts efficacy of", "works well with", "stabilises"
- `ORDER_PHRASES` — phrases indicating order sensitivity: "must be applied before", "apply first", "pH window", "wait between"

`nlp/pubmed_extractor.py`
- Function `load_abstracts_from_xml(xml_path)` — parses XML, extracts abstract text strings, returns list of sentences using spaCy sentence tokeniser
- Function `tag_ingredient_mentions(sentences, ingredient_a, ingredient_b)` — uses spaCy NER + simple string matching to find mentions of both ingredients in the same sentence
- Function `classify_relation(sentence, ingredient_a, ingredient_b)` — checks sentence against `CONFLICT_VERBS` and `SYNERGY_VERBS`. Returns `(RelationType, matched_phrase)` tuple. Falls back to `NEUTRAL` if no signal found.
- Function `extract_triples_from_file(xml_path, ingredient_a, ingredient_b)` — runs full pipeline on one file, returns list of `(ingredient_a, ingredient_b, relation_type, sentence, pmid)` tuples
- Function `run_all()` — iterates all XML files in `data/raw/pubmed_abstracts/`, runs extraction, aggregates triples. Groups by ingredient pair — counts agreements and disagreements per pair. Writes `data/processed/conflict_pairs_raw.csv` and `data/processed/synergy_pairs_raw.csv` with columns: `ingredient_a`, `ingredient_b`, `relation_type`, `source_count`, `agreement_ratio`, `primary_mechanism`, `supporting_pmids`

**Cross-file updates:**
- `nlp/normaliser.py` — call `normaliser.normalise()` on both ingredients before writing to CSV to ensure canonical names

**Exit condition:** Both CSVs exist. `conflict_pairs_raw.csv` has at least 30 rows. Retinol + lactic acid row exists with `relation_type = CONFLICT`.

---

### Section 1.5 — Scraper: EWG Skin Deep

**What it does:** Scrapes EWG Skin Deep for toxicity score and allergen flag for each ingredient in `ingredients_master.csv`. Saves raw HTML to `data/raw/ewg_scraped/`. Enriches `ingredients_master.csv` with `ewg_score` and `pih_risk` columns.

**Files to create:**

`ingestion/ewg_scraper.py`
- Function `build_ewg_url(canonical_inci_name)` — constructs EWG search URL for the ingredient
- Function `fetch_ingredient_page(url, ingredient_name)` — fetches HTML, caches to `data/raw/ewg_scraped/{ingredient_name}.html`. Returns cached file if exists.
- Function `parse_ewg_score(html)` — extracts the hazard score (1–10), allergen flag (bool), and any specific skin tone warnings from parsed HTML
- Function `enrich_master_csv()` — iterates `ingredients_master.csv`, runs scraper for each ingredient, adds `ewg_score` and `allergen_flag` columns, writes back. Sleeps `random.uniform(2,4)` between requests.

**Exit condition:** `ingredients_master.csv` has `ewg_score` values populated for at least 70% of rows.

---

### Section 1.6 — Scraper: INCIDecoder

**What it does:** Scrapes INCIDecoder for function labels (humectant, exfoliant, occlusive, etc.) and irritancy level for each ingredient. Adds `function_labels` and `irritancy_level` to `ingredients_master.csv`.

**Files to create:**

`ingestion/incidecoder_scraper.py`
- Same pattern as EWG scraper: URL builder, page fetcher with cache, parser, enrichment function
- Parser extracts: function label tags (the coloured badges on INCIDecoder pages), irritancy description, skin type warnings
- Maps INCIDecoder function tags to our internal vocabulary (e.g. "skin-identical ingredient" → "skin-restoring")

**Cross-file updates:**
- `data/processed/ingredients_master.csv` — adds `function_labels` (pipe-separated list) and `irritancy_level` columns

**Exit condition:** `ingredients_master.csv` has `function_labels` populated for at least 60% of rows.

---

# PHASE 2 — KNOWLEDGE GRAPH
## "Raw data becomes a queryable, typed, confident graph"

---

### Section 2.1 — PubChem Enrichment

**What it does:** For each ingredient in `ingredients_master.csv`, fetches molecular data from PubChem REST API: CID (compound ID), optimal pH range, known skin safety flags. This is the final enrichment pass before graph build.

**Files to create:**

`ingestion/pubchem_fetcher.py`
- Function `fetch_compound_data(ingredient_name)` — queries PubChem PUG REST API at `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/JSON`. Extracts: `CID`, molecular weight, IUPAC name (used for alias confirmation), any GHS hazard classifications
- Function `fetch_ph_data(cid)` — queries PubChem for pH-related properties. Note: pH data is sparse in PubChem; for most skincare actives we use curated values. Maintain a `PH_DEFAULTS` dict in this file for the top 100 ingredients as fallback.
- Function `enrich_master_csv()` — adds `pubchem_cid` and `ph_optimal` columns to `ingredients_master.csv`. Sleeps 0.2s between calls.

**Exit condition:** `ingredients_master.csv` has `pubchem_cid` populated for at least 80% of known actives.

---

### Section 2.2 — Graph Builder

**What it does:** Assembles the NetworkX knowledge graph from all processed data. Every row in `conflict_pairs_raw.csv` becomes a CONFLICT edge. Every row in `synergy_pairs_raw.csv` becomes a SYNERGY edge. Every row in `ingredients_master.csv` becomes a node. The graph is serialised to disk.

**Files to create:**

`graph/builder.py`
- Function `build_nodes(G, master_csv_path)` — reads `ingredients_master.csv`, creates one `NetworkX` node per ingredient using canonical INCI name as node ID. Sets all node attributes from the `NodeData` schema (ewg_score, function_labels, irritancy_level, ph_optimal, etc.)
- Function `build_conflict_edges(G, conflict_csv_path)` — reads `conflict_pairs_raw.csv`, creates directed edges (A→B) for each conflict pair. Sets edge attributes: `relation_type=CONFLICT`, `base_score` (derived from `agreement_ratio * 0.9 + source_count_normalized * 0.1`), `source_count`, `agreement_ratio`, `primary_mechanism`. Derives `pih_risk` flag: True if mechanism mentions "PIH" or "hyperpigmentation" or both ingredients have high irritancy level.
- Function `build_synergy_edges(G, synergy_csv_path)` — same pattern for SYNERGY edges
- Function `build_order_edges(G)` — a small hardcoded set of known ORDER_SENSITIVE edges (e.g., Vitamin C before niacinamide in AM, SPF always last). These are factual skincare rules, not literature-derived. Hardcode ~15 important ones.
- Function `compute_confidence_tiers(G)` — iterates all edges, sets `confidence_tier`: "high" if source_count >= 10, "medium" if 4–9, "low" if 1–3
- Function `run_build()` — calls all build functions in sequence, writes `data/graph/skingraph_v1.gpickle` and `data/graph/skingraph_v1_metadata.json` (node count, edge count, build timestamp, version string)

**Exit condition:** `data/graph/skingraph_v1.gpickle` exists. Loading it with NetworkX gives a graph with 200+ nodes and 50+ edges. The retinol → lactic_acid CONFLICT edge exists with correct attributes.

---

### Section 2.3 — Graph Query Interface

**What it does:** Provides a clean, typed API over the raw NetworkX graph. No code outside this file should call NetworkX directly. All graph reads go through these functions.

**Files to create:**

`graph/queries.py`
- Imports `nx`, `settings`, and all types from `graph/schema.py`
- `GraphStore` class — loaded once at startup, holds the NetworkX graph in memory
  - `load(graph_path)` — class method, returns a `GraphStore` instance with the graph loaded
  - `get_conflict_pairs(ingredients: List[str]) -> List[EdgeData]` — generates all pairs from the ingredient list (combinatorially), checks each pair for a CONFLICT edge, returns list of matching EdgeData objects
  - `get_synergy_pairs(ingredients: List[str]) -> List[EdgeData]` — same for SYNERGY
  - `get_order_pairs(ingredients: List[str]) -> List[EdgeData]` — same for ORDER_SENSITIVE
  - `get_node(ingredient: str) -> Optional[NodeData]` — returns full NodeData for a single ingredient, or None if not in graph
  - `get_all_edges_for_ingredient(ingredient: str) -> List[EdgeData]` — all edges (any type) where this ingredient is a node
  - `has_node(ingredient: str) -> bool` — simple existence check
  - `get_routine_risk_matrix(ingredients: List[str]) -> RiskMatrix` — calls `get_conflict_pairs` + `get_order_pairs`, assembles into a `RiskMatrix` dataclass

**Exit condition:** `from graph.queries import GraphStore` works. `gs = GraphStore.load(path)` works. `gs.get_conflict_pairs(["retinol", "lactic_acid"])` returns a non-empty list.

---

### Section 2.4 — Confidence Module

**What it does:** Given a graph edge and a skin profile, produces a `ConfidenceReport`. This is a pure computation — no I/O, no side effects.

**Files to create:**

`graph/confidence.py`
- Function `compute_confidence(edge: EdgeData, skin_profile: Optional[SkinProfile]) -> ConfidenceReport`
  - If `skin_profile` is None, use population baseline (no Fitzpatrick penalty, medium sensitivity)
  - `source_confidence = min(edge.source_count / 10, 1.0)` — saturates at 10 sources
  - `agreement_confidence = edge.agreement_ratio`
  - `coverage_confidence = 1.0 if edge.fitzpatrick_iv_v_flag else 0.75` — penalises Western-only data
  - `overall = source_confidence * 0.4 + agreement_confidence * 0.4 + coverage_confidence * 0.2`
  - Derive `uncertainty_note`: if skin profile has Fitzpatrick IV/V/VI and edge has no Fitzpatrick coverage, set note to explain this
  - Return `ConfidenceReport(score=overall, tier=..., source_count=..., uncertainty_note=...)`

**Exit condition:** Unit testable. `compute_confidence(edge, profile)` returns a `ConfidenceReport` with all fields populated. Fitzpatrick IV profile with a non-covered edge returns a non-None `uncertainty_note`.

---

# PHASE 3 — SCORING ENGINE
## "Graph data becomes personalised conflict scores"

---

### Section 3.1 — Base Scorer

**What it does:** Given a list of ingredients (already normalised), retrieves all conflict/synergy/order edges from the graph and returns raw scores. No skin personalisation yet — this is the population-level baseline.

**Files to create:**

`scoring/base_scorer.py`
- `BaseScorer` class
  - Constructor takes a `GraphStore` instance
  - `score_pair(ingredient_a: str, ingredient_b: str) -> Optional[float]` — looks up conflict edge between the pair, returns `edge.base_score` or `None` if no edge exists (no known interaction, not zero risk — unknown)
  - `score_all_pairs(ingredients: List[str]) -> List[Tuple[str, str, float]]` — generates all pairs combinatorially, calls `score_pair` for each, returns list of (a, b, score) tuples, skipping pairs with no edge
  - `get_synergies(ingredients: List[str]) -> List[EdgeData]` — returns all synergy edges
  - `get_order_issues(ingredients: List[str]) -> List[EdgeData]` — returns all order-sensitive edges

**Exit condition:** `scorer.score_pair("retinol", "lactic_acid")` returns a float between 0 and 1.

---

### Section 3.2 — Skin Profile Adjuster

**What it does:** Takes a base score + skin profile and returns a personalised adjusted score. This is where the Fitzpatrick multipliers and sensitivity multipliers are applied.

**Files to create:**

`scoring/skin_adjuster.py`
- Constants: `FITZPATRICK_PIH_MULTIPLIER` dict (I: 1.0 through VI: 1.25), `SENSITIVITY_MULTIPLIER` dict (low: 1.0, moderate: 1.10, high: 1.25)
- Function `adjust_score(base_score: float, edge: EdgeData, skin_profile: SkinProfile) -> float`
  - Apply Fitzpatrick multiplier only when `edge.pih_risk == True`
  - Apply sensitivity multiplier when `edge.relation_type == CONFLICT`
  - Cap at 1.0
  - Return adjusted float
- Function `adjust_all(base_scores: List[Tuple], edges: List[EdgeData], skin_profile: SkinProfile) -> List[Tuple]` — maps adjustment over all pairs
- Function `explain_delta(base_score: float, adjusted_score: float, edge: EdgeData, skin_profile: SkinProfile) -> str` — returns a short human-readable string explaining why the score was elevated (used by the LLM explainer as structured input)

**Exit condition:** For a Fitzpatrick IV profile with high sensitivity, `adjust_score` on a PIH-risk edge produces a score meaningfully higher than base. Score never exceeds 1.0.

---

### Section 3.3 — Combo Scorer

**What it does:** When a user scans 3–5 products, individual pair scores don't capture the total burden. This module aggregates all pair scores into a routine-level risk summary and generates AM/PM sequencing recommendations.

**Files to create:**

`scoring/combo_scorer.py`
- Function `compute_cumulative_burden(adjusted_scores: List[float]) -> float`
  - Not a simple average — uses a diminishing-returns formula: each additional conflict adds less to the total than the previous one, but the total is always higher than any single score
  - Formula: `1 - product((1 - score) for score in scores)` — this is the standard probabilistic OR formula. Gives the probability that at least one conflict manifests.
- Function `classify_overall_risk(burden_score: float) -> str` — maps burden to "safe" / "caution" / "high_risk" using thresholds from `settings`
- Function `build_routine_recommendation(ingredients: List[str], order_edges: List[EdgeData], conflict_pairs: List[Tuple]) -> RoutineOutput`
  - AM routine: Vitamin C, AHAs/BHAs, niacinamide, SPF (always last)
  - PM routine: retinol, heavier actives, moisturiser
  - Any ingredients in a CONFLICT pair get split into AM/PM if possible, or flagged as "do not use together"
  - Returns a `RoutineOutput` with ordered lists for AM and PM

**Exit condition:** `compute_cumulative_burden([0.7, 0.6, 0.5])` returns a value higher than 0.7 but less than 1.0. `build_routine_recommendation` correctly places retinol in PM and Vitamin C in AM.

---

# PHASE 4 — ML MODEL
## "Graph knowledge becomes a trainable, improvable model"
### This phase runs on Google Colab with GPU runtime

---

### Section 4.1 — Feature Builder

**What it does:** For any (ingredient_a, ingredient_b, skin_profile) triple, constructs a fixed-length numeric feature vector that the LightGBM model can train on and predict from. This runs both at training time (on Colab) and at inference time (locally, CPU).

**Files to create:**

`models/feature_builder.py`
- `FEATURE_NAMES` constant — ordered list of all 17 feature names from the architecture doc. Order matters — must be identical at train and inference time.
- Function `build_feature_vector(edge: EdgeData, skin_profile: SkinProfile, node_a: NodeData, node_b: NodeData) -> Dict[str, float]`
  - Numerics: `base_conflict_score`, `skin_adjusted_score`, `source_count` (raw int), `agreement_ratio`
  - One-hot skin type flags: `skin_type_oily`, `skin_type_dry`, `skin_type_combination` (all three 0 if "normal")
  - Fitzpatrick binary flags: `fitzpatrick_iv` (1 if skin_profile.fitzpatrick == "IV"), `fitzpatrick_v_vi` (1 if V or VI)
  - Sensitivity flag: `sensitivity_high` (1 if "high")
  - Concern flags: `concern_acne`, `concern_hyperpigmentation` (from skin_profile.concerns list)
  - Edge flags: `pih_risk_edge` (from edge.pih_risk)
  - Node irritancy flags: `ingredient_a_irritancy_high` (1 if node_a.irritancy_level == "high"), same for B
  - Concentration fields: `concentration_a`, `concentration_b` — default to 0.0 if unknown
  - Returns a dict keyed by `FEATURE_NAMES` — always same length, always same key order
- Function `build_feature_array(feature_dict: Dict) -> np.ndarray` — converts dict to numpy array in the correct column order for the model

**Exit condition:** `build_feature_vector(edge, profile, node_a, node_b)` returns a dict with exactly 17 keys, all numeric values, no None values (defaults used where data is missing).

---

### Section 4.2 — Training Script (Colab)

**What it does:** Loads the graph, builds a training dataset from all conflict/synergy edges with synthetic labels, trains a LightGBM binary classifier, evaluates it, and saves the model. Runs once on Colab, output committed to repo.

**Files to create:**

`models/train_lightgbm.py`
- This script is designed to be uploaded to Colab and run there. It imports from `models/feature_builder.py` and `graph/queries.py` — these must be importable in Colab (upload the whole skingraph directory or install from repo)
- Step 1: Load graph (`GraphStore.load(...)`)
- Step 2: For every edge in the graph with `relation_type == CONFLICT`, generate training samples across 5 synthetic skin profiles (oily Fitzpatrick IV high sensitivity, dry Fitzpatrick II low sensitivity, etc.) — creates ~5x more rows than edges
- Step 3: Call `build_feature_vector` for each (edge, profile) pair → feature matrix X
- Step 4: Build target vector y: `y = skin_adjusted_score + np.random.normal(0, 0.05, size=len(X))`, clipped to [0, 1]. This is the synthetic label strategy.
- Step 5: Train/test split (80/20), train `LGBMClassifier` with `device='gpu'` (falls back to `'cpu'` automatically if no GPU)
- Step 6: Evaluate: log AUC-ROC on test set, print top-10 feature importances
- Step 7: `joblib.dump(model, 'models/artifacts/lgbm_risk_v1.pkl')`
- Step 8: Write training metadata JSON alongside pkl: feature names, training date, AUC score, n_samples

**Exit condition:** `models/artifacts/lgbm_risk_v1.pkl` exists locally (committed from Colab output). Training metadata JSON exists alongside it. AUC-ROC on test set > 0.70.

---

### Section 4.3 — Inference Wrapper

**What it does:** Loads the trained model and exposes a simple predict interface. Instantiated once at API startup. Fast CPU inference.

**Files to create:**

`models/predict.py`
- `RiskModel` class
  - `__init__(model_path: str)` — loads `.pkl` with `joblib.load`, validates feature names match `FEATURE_NAMES` constant from `feature_builder.py`, raises on mismatch (prevents silent version drift)
  - `predict(feature_dict: Dict) -> ModelScore` — converts dict to array, calls `model.predict_proba`, returns `ModelScore` dataclass with `ml_score` (float) and `ml_sample_size` (from stored metadata)
- `ModelScore` dataclass — `ml_score: float`, `ml_sample_size: int`

**Cross-file updates:**
- `graph/schema.py` — add `ModelScore` dataclass here (it's a typed output, belongs with other output types), OR define it in `models/predict.py` and import it from there in `api/schemas.py`. Decision: define in `models/predict.py`, import wherever needed.

**Exit condition:** `model = RiskModel(path); model.predict(feature_dict)` returns a `ModelScore` with a float between 0 and 1.

---

# PHASE 5 — INFERENCE API
## "All components connected behind a clean HTTP interface"

---

### Section 5.1 — API Startup and App Factory

**What it does:** Creates the FastAPI app, loads all heavy resources at startup (graph, model, normaliser), and makes them available to all request handlers via `app.state`. This is the only place where resources are loaded — never per-request.

**Files to create:**

`api/main.py`
- Creates FastAPI app with title, description, version metadata
- `startup` async event handler — loads `GraphStore`, `RiskModel`, `IngredientNormaliser` into `app.state`. Logs node/edge counts on successful graph load. Raises on any load failure (fail fast — don't serve a broken API).
- `shutdown` async event handler — logs shutdown cleanly
- Registers routers from `api/routes/scan.py`, `api/routes/explain.py`, `api/routes/health.py`
- The app object is what `uvicorn` points at: `uvicorn api.main:app`

**Exit condition:** `uvicorn api.main:app --reload` starts without error. Logs show graph loaded with correct node/edge count.

---

### Section 5.2 — Middleware

**What it does:** Adds cross-cutting concerns to every request: request logging, CORS headers (needed when Streamlit calls the API), and basic error handling.

**Files to create:**

`api/middleware.py`
- CORS middleware — allows all origins in development (`*`), will be restricted to brand domains in production
- Request logging middleware — logs method, path, response status, latency for every request
- Global exception handler — catches unhandled exceptions, returns a clean JSON error response instead of a 500 stack trace, logs the full traceback internally

**Cross-file updates:**
- `api/main.py` — add `app.add_middleware(...)` calls and `app.add_exception_handler(...)` after app creation

**Exit condition:** Any request to any endpoint appears in logs. An intentional error in a route returns `{"error": "..."}` JSON, not a raw Python traceback.

---

### Section 5.3 — Health Endpoint

**What it does:** A minimal endpoint that returns system status. Used by monitoring, by the test script to check the API is running, and by the Streamlit app before making scan requests.

**Files to create:**

`api/routes/health.py`
- `GET /health` — returns `{"status": "ok", "graph_nodes": N, "graph_edges": N, "model_loaded": true}`
- Reads counts from `request.app.state.graph` and model

**Exit condition:** `curl http://localhost:8000/health` returns 200 with correct counts.

---

### Section 5.4 — Scan Endpoint (Core)

**What it does:** Implements the full `/scan` request lifecycle from the architecture doc. Orchestrates all previously built modules in the correct order.

**Files to create:**

`api/routes/scan.py`
- `POST /scan` — accepts `ScanRequest`, returns `ScanResponse`
- Full lifecycle (12 steps from architecture doc):
  1. Validate via Pydantic (automatic)
  2. If ingredients not provided in request, look up product name in the graph (partial match on node names) — note: V1 requires ingredient lists to be provided OR looks up known products from Open Beauty Facts. Log a warning if product not found.
  3. Normalise all ingredient names via `request.app.state.normaliser`
  4. Query graph for all pairs
  5. Base score all pairs
  6. Skin-adjust all pairs (use population baseline if no `skin_profile` in request)
  7. Compute confidence for each pair
  8. ML predict for each pair
  9. Ensemble: `final_score = 0.5 * skin_adjusted + 0.3 * ml_score + 0.2 * base_score` (weights from `settings`)
  10. Cumulative burden score
  11. Routine recommendation
  12. Return `ScanResponse` with generated UUID as `scan_id`
- Helper function `build_conflict_result(edge, base, adjusted, ml_score, final, confidence) -> ConflictResult` — assembles the output object, derives `severity` from `final_score` vs thresholds

**Exit condition:** `POST /scan` with a body containing retinol + AHA products returns a `ScanResponse` where the retinol/lactic_acid pair appears in `conflicts` with severity "high".

---

### Section 5.5 — Explain Endpoint

**What it does:** Accepts a `ScanResponse` + `SkinProfile`, calls the LLM explainer, returns plain-language explanation. Separated from `/scan` so that structured data is always fast and free, and LLM is opt-in.

**Files to create:**

`api/routes/explain.py`
- `POST /explain` — accepts `ExplainRequest` (contains embedded `ScanResponse` + `SkinProfile`), returns `ExplainResponse`
- Instantiates `LLMExplainer` using API key from `settings.GROQ_API_KEY`
- Calls `explainer.explain(scan_result, skin_profile)`
- Returns `ExplainResponse(explanation=text, word_count=len(text.split()))`
- If `ANTHROPIC_API_KEY` is not set, returns a 503 with a clear error message

**Exit condition:** `POST /explain` with a valid `ScanResponse` in the body returns a 200 with a 150–250 word explanation that mentions the detected conflict mechanism and the user's skin profile.

---

# PHASE 6 — LLM EXPLAINER
## "Structured scores become honest, personalised, plain-language output"

---

### Section 6.1 — Prompt Builder

**What it does:** Converts a `ScanResponse` and `SkinProfile` into a structured prompt string for Claude. The prompt is deterministic — same inputs always produce the same prompt. It is NOT the LLM's job to discover mechanisms; it only translates pre-computed structured data.

**Files to create:**

`explainer/prompt_builder.py`
- Function `format_conflicts_for_prompt(conflicts: List[ConflictResult]) -> str` — serialises each conflict as a structured text block: pair names, final score, severity, mechanism, confidence tier, uncertainty note. Human-readable but information-dense.
- Function `format_synergies_for_prompt(synergies: List[SynergyResult]) -> str` — same for synergies
- Function `format_confidence_notes(scan_result: ScanResponse) -> str` — aggregates all uncertainty notes into a single section
- Function `build_explain_prompt(scan_result: ScanResponse, skin_profile: SkinProfile) -> str` — assembles the full prompt from the above helpers. Prompt text exactly as defined in architecture doc. Includes the 6 task instructions, language note, and word count constraint.

**Exit condition:** `build_explain_prompt(scan_result, profile)` returns a string with all conflict data embedded and the 6 task instructions present.

---

### Section 6.2 — LLM Client

**What it does:** Wraps the Anthropic API call. Handles retries on rate limit errors, logs token usage, and returns the explanation string.

**Files to create:**

`explainer/llm_client.py`
- `LLMExplainer` class
  - `__init__(api_key: str)` — initialises `anthropic.Anthropic(api_key=api_key)`
  - `explain(scan_result: ScanResponse, skin_profile: SkinProfile) -> str`
    - Builds prompt via `prompt_builder.build_explain_prompt`
    - Calls `client.messages.create` with `model="claude-sonnet-4-20250514"`, `max_tokens=400`
    - Logs input token count and output token count for cost tracking
    - On `anthropic.RateLimitError`: sleeps 10 seconds, retries once. On second failure, raises.
    - Returns `message.content[0].text`

**Exit condition:** `explainer.explain(scan_result, profile)` returns a string of 150–250 words. Calling with an invalid API key raises a clean error, not a silent empty string.

---

# PHASE 7 — TESTING AND SCRIPTS
## "The pipeline is verified end-to-end with known inputs and expected outputs"

---

### Section 7.1 — Unit Tests

**What it does:** Individual module tests. Each test file covers exactly one module. Tests use fixtures (known conflict pairs) to verify correctness, not randomness.

**Files to create:**

`tests/fixtures/known_conflicts.json`
- A JSON array of known conflict pairs: each with `ingredient_a`, `ingredient_b`, `expected_relation`, `expected_severity_range` (e.g., [0.65, 0.95])
- Seed values: retinol + lactic_acid (conflict, high), niacinamide + zinc_pca (synergy), vitamin_c + retinol (conflict, medium-high), spf + moisturiser order issue

`tests/test_normaliser.py`
- Tests that "vitamin a" → "retinol", "vit c" → "ascorbic_acid", a completely unknown string → "UNKNOWN::..." format

`tests/test_graph_queries.py`
- Tests that `get_conflict_pairs(["retinol", "lactic_acid"])` returns non-empty list
- Tests that `get_synergy_pairs(["niacinamide", "zinc_pca"])` returns a SYNERGY edge

`tests/test_scorer.py`
- Tests that `base_scorer.score_pair("retinol", "lactic_acid")` returns float in range [0.6, 1.0]
- Tests that `skin_adjuster.adjust_score` with Fitzpatrick IV returns a higher score than the same call with Fitzpatrick I (for a PIH-risk edge)
- Tests that `combo_scorer.compute_cumulative_burden([0.7, 0.6])` > 0.7

`tests/test_api.py`
- Uses `httpx.AsyncClient` pointed at the FastAPI test app
- Tests `GET /health` returns 200
- Tests `POST /scan` with retinol + AHA payload returns 200 with at least one conflict

**Exit condition:** `pytest tests/` runs without failures.

---

### Section 7.2 — End-to-End Test Script

**What it does:** A CLI script that exercises the full pipeline from raw product name input to final scored output. This is the primary developer tool — run it after every sprint and after any significant change.

**Files to create:**

`scripts/test_scan.py`
- Accepts command-line arguments: product names (2–5), optional skin profile flags
- Example: `python scripts/test_scan.py "Minimalist Retinol 0.6%" "Deconstruct AHA 25%" --skin-type oily --fitzpatrick IV --sensitivity high`
- Flow: parse args → construct `ScanRequest` → POST to running API at `localhost:8000/scan` → pretty-print the `ScanResponse`
- Pretty-print format: colour-coded (use `colorama` or just ASCII separators), shows each conflict with score, severity, mechanism, confidence tier and uncertainty note, then shows routine recommendation
- If API is not running, prints a clear error and exits

`scripts/run_ingestion.py`
- CLI to run the full data pipeline from scratch
- Flags: `--skip-download` (uses cached raw files), `--skip-scraping` (skips EWG/INCIDecoder), `--pairs-only` (only re-runs PubMed for new pairs)
- Calls all fetchers and scrapers in correct order, logs progress

`scripts/build_graph.py`
- CLI to rebuild the graph from processed CSVs
- Useful when you've manually edited a CSV (e.g., adding a curated conflict pair) and want to rebuild without re-running ingestion
- Calls `graph.builder.run_build()`

**Exit condition:** Running the test script with retinol + AHA payload prints a conflict analysis with severity "high" and a routine recommendation placing retinol in PM.

---

# PHASE 8 — FINAL WIRING AND VALIDATION
## "Everything connected, nothing broken, ready for Streamlit"

---

### Section 8.1 — Integration Validation

**What it does:** A final checklist pass verifying that all modules integrate correctly with no import cycles, no hardcoded values outside `settings.py`, and no raw dicts where typed dataclasses should be used.

**No new files.** Only verification:
- Run `python -c "from api.main import app"` — if this imports without error, the full dependency tree is clean
- Run `pytest tests/` — all tests pass
- Run `python scripts/test_scan.py "Minimalist Retinol 0.6%" "Deconstruct AHA 25%"` — produces expected output
- Run `python scripts/test_scan.py "Minimalist Niacinamide 10%" "Zinc PCA serum"` — produces a synergy result
- Grep codebase for hardcoded floats like `0.75` outside `config/settings.py` — none should exist

**Exit condition:** All 4 checks above pass. The system is ready for a Streamlit frontend to be built on top of the `/scan` and `/explain` endpoints.

---

### Section 8.2 — README

**What it does:** Documents the system for any developer who comes to it cold, or for any AI agent re-entering the project after a break.

**Files to create:**

`README.md`
- Project overview (2 paragraphs)
- System architecture diagram (ASCII version of the flow from architecture doc)
- Setup instructions: clone, create `.env` from `.env.example`, `pip install -r requirements.txt`, run ingestion, build graph, start API
- How to run the test script
- How to train the model on Colab (link to `requirements_colab.txt`, explain upload steps)
- Module reference: one line per file explaining what it does
- Sprint log: date and what was built per sprint (keep this updated)

**Exit condition:** A developer with no prior context can set up and run the system by following README alone.

---

# QUICK REFERENCE — FULL FILE LIST IN ORDER OF CREATION

```
Phase 0:  .gitignore, .env.example, requirements.txt, requirements_colab.txt
          config/settings.py, config/logging_config.py
          graph/schema.py
          api/schemas.py

Phase 1:  ingestion/openbf_fetcher.py
          nlp/normaliser.py
          ingestion/pubmed_fetcher.py
          nlp/relation_types.py, nlp/pubmed_extractor.py
          ingestion/ewg_scraper.py
          ingestion/incidecoder_scraper.py

Phase 2:  ingestion/pubchem_fetcher.py
          graph/builder.py
          graph/queries.py
          graph/confidence.py

Phase 3:  scoring/base_scorer.py
          scoring/skin_adjuster.py
          scoring/combo_scorer.py

Phase 4:  models/feature_builder.py
          models/train_lightgbm.py  [RUNS ON COLAB]
          models/predict.py

Phase 5:  api/main.py
          api/middleware.py
          api/routes/health.py
          api/routes/scan.py
          api/routes/explain.py

Phase 6:  explainer/prompt_builder.py
          explainer/llm_client.py

Phase 7:  tests/fixtures/known_conflicts.json
          tests/test_normaliser.py
          tests/test_graph_queries.py
          tests/test_scorer.py
          tests/test_api.py
          scripts/test_scan.py
          scripts/run_ingestion.py
          scripts/build_graph.py

Phase 8:  README.md
```

---

*This plan is complete. Each section is executable independently. No section has an undeclared upstream dependency. The AI agent executing this plan should process one section at a time, verify the exit condition before proceeding, and update this document's sprint log in the README as sections complete.*