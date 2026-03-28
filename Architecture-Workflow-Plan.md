# SkinGraph — Cosmetic Ingredient Conflict Detection System
## Full Production Architecture & Development Workflow
### Version: V1 (MVP) → V2 (ML Layer) → V3 (Platform)

---

## 0. DOCUMENT PURPOSE

This document is the single source of truth for system design, data flow, module ownership, and build order. Every file, pipeline, and API call is specified here before a single line of code is written. Read this entirely before touching any code. This is a living spec — update it as decisions change.

---

## 1. THE PROBLEM WE ARE SOLVING (Technical Framing)

You are building a **personalised ingredient safety inference engine** — not a lookup table, not a FAQ bot. The distinction matters for every architectural decision that follows.

A lookup table says: "Retinol + AHA = bad." Full stop.

Your system says: "For your specific skin profile (oily, Fitzpatrick IV, high sensitivity, PIH history), combining retinol at 0.6% and lactic acid at 5% carries a conflict score of 0.91, elevated from the population baseline of 0.72, based on 14 peer-reviewed sources, with limited Fitzpatrick IV coverage in the literature. Here is why, here is the sequencing fix, and here is what to watch for in 2 weeks."

That's not a lookup. That's a reasoning system. The architecture has to support that.

---

## 2. HIGH-LEVEL SYSTEM OVERVIEW

```
RAW DATA SOURCES
     │
     ▼
[Data Ingestion Pipeline]  ← runs once, then on schedule
     │
     ▼
[Knowledge Graph]  ← NetworkX graph, stored as .gpickle
     │
     ├──► [NLP Extraction Module]  ← PubMed abstracts → edge relationships
     │
     ▼
[Conflict Scoring Engine]  ← base scores + skin profile adjustment
     │
     ▼
[Confidence Layer]  ← source count, agreement ratio, Fitzpatrick coverage
     │
     ├──► [LightGBM Risk Model]  ← trained on Colab, loaded as .pkl
     │
     ▼
[Inference API]  ← FastAPI, stateless, JSON in/out
     │
     ▼
[Output Formatter + LLM Explainer]  ← Groq API (Llama-3.3-70b-versatile)
     │
     ▼
[Test Script / Streamlit UI]  ← user-facing layer
```

Everything above the Inference API is **offline / batch**. Everything at and below is **online / real-time**.

---

## 3. DIRECTORY STRUCTURE

```
skingraph/
│
├── data/
│   ├── raw/                    # downloaded dumps, never modified
│   │   ├── openbf_products.json
│   │   ├── pubmed_abstracts/   # .xml files per query
│   │   └── ewg_scraped/        # scraped HTML, cached
│   ├── processed/              # cleaned, normalised
│   │   ├── ingredients_master.csv      # canonical INCI name + aliases
│   │   ├── conflict_pairs_raw.csv      # extracted from NLP + scraping
│   │   └── synergy_pairs_raw.csv
│   └── graph/
│       ├── skingraph_v1.gpickle        # main knowledge graph
│       └── skingraph_v1_metadata.json  # build date, node/edge counts
│
├── ingestion/
│   ├── __init__.py
│   ├── openbf_fetcher.py       # fetches Open Beauty Facts dump
│   ├── pubmed_fetcher.py       # fetches PubMed abstracts via Entrez API
│   ├── ewg_scraper.py          # scrapes EWG Skin Deep
│   ├── incidecoder_scraper.py  # scrapes INCIDecoder
│   └── cosdna_scraper.py       # scrapes CosDNA
│
├── nlp/
│   ├── __init__.py
│   ├── normaliser.py           # raw name → canonical INCI name
│   ├── pubmed_extractor.py     # NLP pipeline: abstract → (ing_A, ing_B, relation)
│   └── relation_types.py       # enum: CONFLICT, SYNERGY, ORDER, NEUTRAL
│
├── graph/
│   ├── __init__.py
│   ├── builder.py              # assembles graph from all processed data
│   ├── schema.py               # node/edge schema definitions
│   ├── queries.py              # reusable graph query functions
│   └── confidence.py           # source count, agreement, Fitzpatrick coverage
│
├── scoring/
│   ├── __init__.py
│   ├── base_scorer.py          # graph-lookup-based conflict score
│   ├── skin_adjuster.py        # modulates score by skin profile vector
│   └── combo_scorer.py         # scores a full product set, not just pairs
│
├── models/
│   ├── __init__.py
│   ├── train_lightgbm.py       # training script — runs on Colab
│   ├── feature_builder.py      # builds feature vector for each pair + skin profile
│   ├── predict.py              # loads .pkl and runs inference
│   └── artifacts/
│       └── lgbm_risk_v1.pkl    # trained model (committed after Colab run)
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entrypoint
│   ├── routes/
│   │   ├── scan.py             # POST /scan — core endpoint
│   │   ├── explain.py          # POST /explain — LLM explanation
│   │   └── health.py           # GET /health
│   ├── schemas.py              # Pydantic request/response models
│   └── middleware.py           # rate limiting, CORS, logging
│
├── explainer/
│   ├── __init__.py
│   ├── llm_client.py           # wrapper around Groq API
│   └── prompt_builder.py       # builds structured prompt from conflict output
│
├── tests/
│   ├── test_normaliser.py
│   ├── test_scorer.py
│   ├── test_api.py
│   └── fixtures/               # sample inputs/outputs for tests
│
├── scripts/
│   ├── run_ingestion.py        # CLI: run full data pipeline
│   ├── build_graph.py          # CLI: build graph from processed data
│   ├── test_scan.py            # CLI: test the scan endpoint manually
│   └── evaluate_model.py       # CLI: run accuracy benchmarks
│
├── config/
│   ├── settings.py             # centralised config (paths, thresholds, API keys)
│   └── logging_config.py       # structured logging setup
│
├── requirements.txt
├── requirements_colab.txt      # GPU-specific deps for Colab training
├── .env.example
└── README.md
```

Every module has a single responsibility. Every shared utility lives in its own file. Nothing is duplicated.

---

## 4. DATA LAYER — SOURCES, SCHEMAS, INGESTION

### 4.1 Source Inventory

| Source | What we get | How | Priority |
|--------|------------|-----|---------|
| Open Beauty Facts | 140k product → ingredient lists | Free JSON dump | P0 |
| PubMed | 35M+ abstracts | Free Entrez REST API | P0 |
| PubChem | Molecular data, pH, safety flags | Free REST API | P0 |
| EWG Skin Deep | Toxicity score, allergen flag | Scrape (respectful rate) | P1 |
| INCIDecoder | Function labels, irritancy level | Scrape | P1 |
| CosDNA | Acne risk, interaction ratings | Scrape | P2 |

All scraping uses `requests` + `BeautifulSoup4` with a 2–3 second delay between requests and a rotating `User-Agent` header. We cache raw HTML to `data/raw/` so we never re-scrape the same page.

### 4.2 The Master Ingredient List

This is the most important asset in the system. Every ingredient from every source gets normalised to a **canonical INCI name** before it touches the graph.

**`ingredients_master.csv` schema:**
```
canonical_inci_name | aliases | pubchem_cid | ewg_score | function_labels | irritancy_level | acne_risk_score
retinol | vitamin a, retinyl, retinol usp | 445354 | 7 | cell-communicating | high | moderate
niacinamide | vitamin b3, nicotinamide | 936 | 1 | skin-restoring | low | none
```

**`normaliser.py`** is the gatekeeper. Every ingredient name that enters the system — from user input, from scraped data, from PubMed — runs through this module before anything else. It uses:
1. Exact match against the alias column
2. Fuzzy match (`rapidfuzz`, threshold 85) for typos and brand names
3. If no match: PubChem synonym lookup via API
4. If still no match: flag as `UNKNOWN`, log for review

This normaliser is used by ingestion, by the API at inference time, and by the test script. One function, called everywhere.

### 4.3 PubMed NLP Pipeline

This is where edge relationships come from. The pipeline in `nlp/pubmed_extractor.py`:

1. Query PubMed using Entrez API for each priority ingredient pair (e.g., "retinol lactic acid skin interaction")
2. Fetch abstracts in batches of 100 (free, no auth required — just add your email to the `Entrez.email` field per NCBI guidelines)
3. Run each abstract through a **relation extraction pipeline**:
   - Sentence tokenisation (spaCy)
   - Named entity recognition — tag ingredient mentions
   - Dependency parse to find relation verb between entity pairs
   - Classify relation as: CONFLICT / SYNERGY / ORDER_SENSITIVE / NEUTRAL / UNCLEAR
4. Each extracted triple `(ingredient_A, ingredient_B, relation, sentence, pmid)` is written to `conflict_pairs_raw.csv`
5. Multiple extractions for the same pair are **aggregated**: count of supporting sentences becomes the `source_count` edge attribute

For V1, the NLP classification can use a rule-based approach with a curated verb/phrase list (e.g., "irritates," "degrades," "conflicts," "enhances," "potentiates"). For V2, fine-tune a small BERT model on this.

**Rate limits:** NCBI Entrez allows 3 requests/second without API key, 10/second with (free key). Always use the free key. The fetcher in `pubmed_fetcher.py` handles throttling automatically.

---

## 5. KNOWLEDGE GRAPH — SCHEMA AND STRUCTURE

### 5.1 Why a Graph

A graph is the right data structure here because:
- Ingredient interactions are **relational** — the unit of analysis is a pair (or triple), not a single node
- The same ingredient appears in dozens of relationships — the graph structure avoids data duplication
- Graph traversal lets you answer questions like "what is the safest path through these 5 ingredients" — a table cannot do that

We use **NetworkX** — pure Python, no database server required, serialises to `.gpickle`, fast enough for 10k-node graphs on CPU.

### 5.2 Node Schema

```python
# graph/schema.py
NODE_SCHEMA = {
    "canonical_inci": str,          # primary key
    "pubchem_cid": int,
    "function_labels": List[str],   # ["humectant", "exfoliant"]
    "irritancy_level": str,         # "low" | "moderate" | "high"
    "acne_risk": str,               # "none" | "low" | "moderate" | "high"
    "ewg_score": int,               # 1-10
    "ph_optimal": Tuple[float, float],  # (min_ph, max_ph)
    "skin_type_caution": List[str], # ["sensitive", "rosacea"]
    "fitzpatrick_notes": str,       # any specific notes for darker skin tones
}
```

### 5.3 Edge Schema

```python
EDGE_SCHEMA = {
    "relation_type": str,           # "CONFLICT" | "SYNERGY" | "ORDER_SENSITIVE" | "NEUTRAL"
    "base_score": float,            # 0.0–1.0, population-level conflict/synergy strength
    "source_count": int,            # number of supporting literature extractions
    "agreement_ratio": float,       # fraction of sources that agree on the relation
    "primary_mechanism": str,       # "pH incompatibility" | "barrier disruption" | etc.
    "fitzpatrick_iv_v_flag": bool,  # does this edge have specific data for Fitzpatrick IV-V?
    "pih_risk": bool,               # post-inflammatory hyperpigmentation escalation risk
    "confidence_tier": str,         # "high" | "medium" | "low" — derived from source_count
    "sources": List[str],           # pmids or URLs
    "safe_at_concentrations": Optional[str],  # e.g., "safe if AHA < 5%"
    "order_note": Optional[str],    # e.g., "Apply Vitamin C AM, retinol PM only"
}
```

### 5.4 Graph Build Process (`graph/builder.py`)

```
1. Load ingredients_master.csv → create all nodes
2. Load conflict_pairs_raw.csv → create CONFLICT edges with source_count, agreement_ratio
3. Load synergy_pairs_raw.csv → create SYNERGY edges
4. Fetch PubChem pH data for all nodes → annotate pH_optimal
5. Pull EWG scores → annotate ewg_score, pih_risk
6. Pull INCIDecoder function labels → annotate function_labels
7. Compute confidence_tier per edge:
   - source_count >= 10: "high"
   - source_count 4-9: "medium"
   - source_count 1-3: "low"
8. Serialise: nx.write_gpickle(G, "data/graph/skingraph_v1.gpickle")
9. Write metadata JSON: node count, edge count, build timestamp, version
```

### 5.5 Graph Queries (`graph/queries.py`)

Every part of the system that needs to read the graph goes through this module. Functions:

```python
def get_conflict_pairs(ingredients: List[str]) -> List[ConflictEdge]
def get_synergy_pairs(ingredients: List[str]) -> List[SynergyEdge]
def get_order_sensitive_pairs(ingredients: List[str]) -> List[OrderEdge]
def get_ingredient_metadata(ingredient: str) -> NodeData
def get_all_conflicts_for_ingredient(ingredient: str) -> List[ConflictEdge]
def compute_routine_risk_matrix(ingredients: List[str]) -> RiskMatrix
```

These are pure functions. They take ingredient names, return typed data structures. They have no side effects. They are unit-testable.

---

## 6. SCORING ENGINE

### 6.1 Base Scorer (`scoring/base_scorer.py`)

For a given pair `(A, B)`, base conflict score = `edge.base_score`. If no edge exists, score = 0.0 (unknown / no documented interaction).

### 6.2 Skin Profile Adjuster (`scoring/skin_adjuster.py`)

This is the differentiated piece. The skin profile vector:

```python
@dataclass
class SkinProfile:
    skin_type: str          # "oily" | "dry" | "combination" | "normal"
    fitzpatrick: str        # "I" through "VI"
    sensitivity: str        # "low" | "moderate" | "high"
    concerns: List[str]     # ["acne", "hyperpigmentation", "redness"]
    allergies: List[str]    # known allergens
    history: List[str]      # ["retinol-reaction-2023"]
```

The adjuster applies multipliers to the base score:

```python
FITZPATRICK_PIH_MULTIPLIER = {
    "I": 1.0, "II": 1.0, "III": 1.05,
    "IV": 1.15, "V": 1.20, "VI": 1.25
}
# Applied when edge.pih_risk == True

SENSITIVITY_MULTIPLIER = {
    "low": 1.0, "moderate": 1.10, "high": 1.25
}
# Applied when edge.relation_type == "CONFLICT"
```

The adjusted score is capped at 1.0. The delta between base and adjusted is surfaced in the output as an explanation ("Score elevated for your skin profile because...").

### 6.3 Combo Scorer (`scoring/combo_scorer.py`)

When a user scans 3–5 products, we don't just check pairs — we check all pairs, then apply a **cumulative burden score**: if a routine has 3 moderate conflicts, the total skin burden is higher than any single pair implies. This is the basis for the routine recommendation ("even though each pair is medium risk, together they overload the skin barrier").

---

## 7. CONFIDENCE LAYER (`graph/confidence.py`)

This module decorates every score output with honesty metadata:

```python
def compute_confidence(
    edge: EdgeData,
    skin_profile: SkinProfile
) -> ConfidenceReport:
    
    source_confidence = min(edge.source_count / 10, 1.0)  # saturates at 10 sources
    agreement_confidence = edge.agreement_ratio
    coverage_confidence = 1.0 if edge.fitzpatrick_iv_v_flag else 0.75  # penalise Western-only data
    
    overall = (source_confidence * 0.4 + agreement_confidence * 0.4 + coverage_confidence * 0.2)
    
    uncertainty_note = None
    if not edge.fitzpatrick_iv_v_flag and skin_profile.fitzpatrick in ["IV", "V", "VI"]:
        uncertainty_note = "Most studies used Fitzpatrick I-III. Score extrapolated for your skin tone."
    
    return ConfidenceReport(
        score=round(overall, 2),
        tier="high" if overall > 0.75 else "medium" if overall > 0.50 else "low",
        source_count=edge.source_count,
        uncertainty_note=uncertainty_note
    )
```

This function is called on every edge in the output. No conflict or synergy is reported without a confidence score attached. This is not optional — it is structural.

---

## 8. ML MODEL — LIGHTGBM RISK LAYER

### 8.1 Why a Second Model

The knowledge graph gives us the best available static answer. The LightGBM model gives us a **data-corrected answer** based on actual user reactions. For Indian skin types specifically, the model will eventually outperform the graph because Western literature is systematically underrepresented for Fitzpatrick IV-VI. But in V1, this model is trained on synthetic data seeded from graph scores. Real feedback data comes in V2.

### 8.2 Feature Vector (`models/feature_builder.py`)

```python
FEATURES = [
    "base_conflict_score",          # from graph
    "skin_adjusted_score",          # from skin_adjuster
    "source_count",                 # edge attribute
    "agreement_ratio",              # edge attribute
    "skin_type_oily",               # one-hot
    "skin_type_dry",
    "skin_type_combination",
    "fitzpatrick_iv",               # binary flag
    "fitzpatrick_v_vi",             # binary flag
    "sensitivity_high",             # binary flag
    "concern_acne",                 # binary flag
    "concern_hyperpigmentation",    # binary flag
    "pih_risk_edge",                # edge attribute
    "ingredient_a_irritancy_high",  # node attribute
    "ingredient_b_irritancy_high",  # node attribute
    "concentration_a",              # if known (optional, can be null)
    "concentration_b",
]
TARGET = "reaction_probability"     # 0.0–1.0 (V1: derived from graph score, V2: real feedback)
```

### 8.3 Training on Colab (`models/train_lightgbm.py`)

The training script is self-contained and runs on Colab with GPU runtime. Steps:

1. Load `conflict_pairs_raw.csv` + graph edge attributes
2. Build feature matrix using `feature_builder.py`
3. For V1: synthetic labels = `skin_adjusted_score + gaussian_noise(0, 0.05)` — this bootstraps the model on graph knowledge
4. Train LightGBM with `device='gpu'` (or `'cpu'` fallback)
5. Evaluate: AUC-ROC on holdout split, feature importance plot
6. Save: `models/artifacts/lgbm_risk_v1.pkl` using `joblib.dump`
7. Log: training accuracy, feature importances, timestamp

The `.pkl` file is committed to the repo and loaded locally at inference time. No GPU needed for inference — LightGBM prediction on CPU is milliseconds.

### 8.4 Inference (`models/predict.py`)

```python
class RiskModel:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
    
    def predict(self, feature_vector: Dict) -> ModelScore:
        X = build_feature_array(feature_vector)
        prob = self.model.predict_proba(X)[0][1]
        return ModelScore(
            ml_score=round(float(prob), 3),
            ml_sample_size=self.model.n_estimators  # proxy for training size in V1
        )
```

This class is instantiated **once at API startup** and reused for all requests. No per-request model loading.

---

## 9. INFERENCE API (`api/`)

### 9.1 Framework: FastAPI

FastAPI because: async-native, auto-generates OpenAPI docs, Pydantic validation built in, production-grade with uvicorn, and works perfectly with Streamlit as a backend.

### 9.2 Core Endpoint: `POST /scan`

**Request schema (`api/schemas.py`):**
```python
class ScanRequest(BaseModel):
    products: List[ProductInput]  # list of product names or ingredient lists
    skin_profile: Optional[SkinProfile]  # if None, use population baseline

class ProductInput(BaseModel):
    name: str                          # "Minimalist Retinol 0.6%"
    ingredients: Optional[List[str]]   # if known
    routine_step: Optional[str]        # "AM" | "PM" | "both"
```

**Response schema:**
```python
class ScanResponse(BaseModel):
    scan_id: str                        # UUID for logging
    conflicts: List[ConflictResult]
    synergies: List[SynergyResult]
    order_issues: List[OrderResult]
    routine_recommendation: RoutineOutput
    overall_risk_level: str             # "safe" | "caution" | "high_risk"
    cumulative_burden_score: float

class ConflictResult(BaseModel):
    ingredient_a: str
    ingredient_b: str
    product_a: str
    product_b: str
    base_score: float
    skin_adjusted_score: float
    ml_score: float
    final_score: float                  # weighted ensemble of base + ml
    severity: str                       # "low" | "medium" | "high"
    mechanism: str
    confidence: ConfidenceReport
    recommendation: str                 # short action text
```

### 9.3 Request Lifecycle

```
POST /scan
    │
    ├── 1. Validate request (Pydantic)
    ├── 2. Normalise all ingredient names (normaliser.py)
    ├── 3. Load graph (cached at startup)
    ├── 4. query_graph: get all pairs for ingredient list
    ├── 5. base_scorer: score each pair
    ├── 6. skin_adjuster: personalise scores (if skin_profile provided)
    ├── 7. confidence: compute confidence per pair
    ├── 8. risk_model.predict: ML score per pair
    ├── 9. ensemble: final_score = 0.5 * skin_adjusted + 0.3 * ml_score + 0.2 * base
    ├── 10. combo_scorer: cumulative routine burden
    ├── 11. build routine_recommendation (AM/PM sequencing)
    └── 12. return ScanResponse
```

### 9.4 Startup: Resource Loading

```python
# api/main.py
@app.on_event("startup")
async def startup():
    app.state.graph = nx.read_gpickle("data/graph/skingraph_v1.gpickle")
    app.state.risk_model = RiskModel("models/artifacts/lgbm_risk_v1.pkl")
    app.state.normaliser = IngredientNormaliser("data/processed/ingredients_master.csv")
    logger.info(f"Graph loaded: {app.state.graph.number_of_nodes()} nodes, {app.state.graph.number_of_edges()} edges")
```

Graph, model, and normaliser are loaded once and held in app state. Every request reads from memory — no disk I/O per request.

---

## 10. LLM EXPLAINER (`explainer/`)

### 10.1 When it's called

The LLM layer is called **after** the scoring engine has produced its structured output. It receives the full `ScanResponse` and returns a plain-language explanation. It does NOT make any decisions — it only translates structured output into human text. This is critical: the LLM is a formatter, not an oracle.

### 10.2 Prompt Structure (`explainer/prompt_builder.py`)

```python
def build_explain_prompt(scan_result: ScanResponse, skin_profile: SkinProfile) -> str:
    return f"""
You are a skincare science explainer for Indian consumers. You speak clearly, without jargon.
You do not make up information. You only use the data provided below.

USER SKIN PROFILE:
- Skin type: {skin_profile.skin_type}
- Fitzpatrick tone: {skin_profile.fitzpatrick}
- Sensitivity: {skin_profile.sensitivity}
- Concerns: {', '.join(skin_profile.concerns)}

CONFLICT ANALYSIS RESULTS:
{format_conflicts_for_prompt(scan_result.conflicts)}

CONFIDENCE NOTES:
{format_confidence_notes(scan_result)}

TASK:
1. Explain each conflict in 2-3 sentences. Use the mechanism provided. Do not invent mechanisms.
2. If the confidence is low, say so explicitly ("We're less certain about this one — limited studies for your skin type").
3. Give a clear routine recommendation (AM vs PM split, what to avoid entirely).
4. End with one sentence acknowledging any uncertainty.
5. Write for someone who is educated but not a chemist. Think IIT student, not dermatologist.
6. NEVER recommend a specific product. Recommend ingredient patterns only.

LANGUAGE: English. You may use common Indian English phrasing naturally.
OUTPUT LENGTH: 150-250 words.
"""
```

### 10.3 API Client (`explainer/llm_client.py`)

```python
class LLMExplainer:
    def __init__(self, api_key: str):
        self.client = groq.Groq(api_key=api_key)
    
    def explain(self, scan_result: ScanResponse, skin_profile: SkinProfile) -> str:
        prompt = build_explain_prompt(scan_result, skin_profile)
        chat_completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return chat_completion.choices[0].message.content
```

The LLM is called only when the user explicitly requests an explanation (separate `POST /explain` endpoint). The `POST /scan` endpoint returns structured data only — fast, deterministic, no API cost. The explanation is an optional layer on top.

---

## 11. CONFIGURATION AND SECRETS (`config/settings.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Paths
    GRAPH_PATH: str = "data/graph/skingraph_v1.gpickle"
    MODEL_PATH: str = "models/artifacts/lgbm_risk_v1.pkl"
    INGREDIENTS_CSV: str = "data/processed/ingredients_master.csv"
    
    # API keys (loaded from .env)
    GROQ_API_KEY: str
    ENTREZ_EMAIL: str           # required by NCBI, not a secret
    PUBCHEM_API_KEY: str = ""   # optional, increases rate limit
    
    # Scoring thresholds
    CONFLICT_HIGH_THRESHOLD: float = 0.75
    CONFLICT_MEDIUM_THRESHOLD: float = 0.45
    CONFIDENCE_HIGH_THRESHOLD: float = 0.75
    
    # Model ensemble weights
    WEIGHT_SKIN_ADJUSTED: float = 0.5
    WEIGHT_ML_SCORE: float = 0.3
    WEIGHT_BASE_SCORE: float = 0.2
    
    class Config:
        env_file = ".env"

settings = Settings()
```

All hardcoded numbers live here. If a threshold changes, it changes in one place. Every module imports from `config.settings`.

---

## 12. BUILD ORDER — EXACTLY WHAT TO BUILD FIRST

This is the development sequence. Do not skip steps or reorder.

### Sprint 0 — Foundations (Day 1–2)
1. Set up directory structure
2. Write `config/settings.py` and `config/logging_config.py`
3. Write `graph/schema.py` — define all node/edge types as dataclasses
4. Write `api/schemas.py` — define all Pydantic request/response models
5. Write `requirements.txt`

**Why first:** Everything else imports from these files. Getting the types right upfront prevents refactoring later.

### Sprint 1 — Data Ingestion (Day 3–5)
1. `ingestion/openbf_fetcher.py` — download Open Beauty Facts dump
2. `nlp/normaliser.py` — build the master ingredient list
3. `ingestion/pubmed_fetcher.py` — pull abstracts for top 50 ingredient pairs
4. `nlp/pubmed_extractor.py` — extract conflict/synergy triples
5. Manual review: audit 50 extracted triples for accuracy

**Output:** `data/processed/ingredients_master.csv`, `data/processed/conflict_pairs_raw.csv`

### Sprint 2 — Graph (Day 6–8)
1. `graph/builder.py` — assemble graph from processed CSVs
2. `graph/queries.py` — implement all query functions
3. `graph/confidence.py` — implement confidence computation
4. Write `tests/test_graph_queries.py` — test with known pairs
5. Run builder, inspect graph in a notebook: verify retinol–AHA edge exists with correct attributes

**Output:** `data/graph/skingraph_v1.gpickle`

### Sprint 3 — Scoring Engine (Day 9–11)
1. `scoring/base_scorer.py`
2. `scoring/skin_adjuster.py`
3. `scoring/combo_scorer.py`
4. `tests/test_scorer.py` — test known conflict pairs, verify scores in expected range
5. Write `scripts/test_scan.py` — CLI that takes a product list and prints all scores

**This is the first end-to-end test.** Run: `python scripts/test_scan.py "Minimalist Retinol 0.6%" "Deconstruct AHA 25%"` and verify the output makes sense.

### Sprint 4 — ML Model (Day 12–14, on Colab)
1. `models/feature_builder.py`
2. `models/train_lightgbm.py` — run on Colab, generate `.pkl`
3. `models/predict.py` — inference wrapper
4. Commit `models/artifacts/lgbm_risk_v1.pkl` to repo
5. `tests/test_model_predict.py` — verify prediction shape and range

### Sprint 5 — API (Day 15–17)
1. `api/main.py` — startup, resource loading
2. `api/routes/scan.py` — implement `POST /scan`
3. `api/routes/health.py` — basic health check
4. `api/middleware.py` — logging, CORS
5. `tests/test_api.py` — test scan endpoint with sample requests
6. Run: `uvicorn api.main:app --reload` and hit the endpoint with curl

### Sprint 6 — LLM Explainer (Day 18–19)
1. `explainer/prompt_builder.py`
2. `explainer/llm_client.py`
3. `api/routes/explain.py` — `POST /explain`
4. Test: verify explanations cite the mechanism, flag low confidence correctly, stay under 250 words

### Sprint 7 — Test Script (Day 20)
1. `scripts/test_scan.py` — polished CLI that exercises the full pipeline end-to-end
2. Test 10 known conflict/synergy pairs and verify outputs against expected results
3. Document results in `tests/fixtures/expected_outputs.json`

---

## 13. API COST AND RESOURCE MANAGEMENT

**Free resources used:**
- Open Beauty Facts: free JSON dump, no rate limit
- PubMed Entrez: free, 3 req/sec without key, 10/sec with free key
- PubChem REST: free, 5 req/sec
- EWG/INCIDecoder/CosDNA: scraped with respectful rate limiting — no cost
- NetworkX: in-memory, no cost
- LightGBM: inference on CPU, no cost
- Colab GPU: free tier sufficient for training on ~10k samples

**Paid resources:**
- Groq API (Groq): only called for `/explain` endpoint. At ~300 tokens per explanation with Sonnet pricing, this is the only cost in the system. For V1/testing, this is negligible. For production, this is billed per brand API call — pass-through or absorb in SaaS pricing.

**Rate limiting strategy:**
- PubMed: use `time.sleep(0.34)` between calls (3/sec limit), burst to 10/sec with free API key
- PubChem: `time.sleep(0.2)` between calls
- Scraping: `time.sleep(random.uniform(2, 4))` between page fetches
- All fetchers cache to disk on first run — subsequent runs are instant

---

## 14. WHAT V1 DOES NOT INCLUDE (AND WHY)

These are deliberate omissions, not oversights:

- **Streamlit UI:** Frontend is last. The system works via test script and API first. UI is cosmetic at this stage.
- **User accounts / saved routines:** Phase 2 feature. V1 is stateless — every scan is independent.
- **Real feedback loop:** Phase 2. V1 ML model uses synthetic labels. Real reaction data collection is a product decision.
- **WhatsApp bot:** Phase 2. Requires separate bot infra.
- **Brand analytics dashboard:** Phase 3. Requires multi-tenant architecture.
- **Competitor substitution engine:** Phase 2. Requires brand product catalogue integration.

V1 proves one thing: **given a list of products and a skin profile, the system produces a credible, sourced, personalised conflict analysis.** That is the unit of value. Everything else is built on top of that proof.

---

## 15. TESTING PHILOSOPHY

Every module has a corresponding test file. Tests run with `pytest`. Fixtures for known conflict pairs (retinol + AHA, Vitamin C + retinol, niacinamide + zinc) serve as regression tests — if a code change breaks these expected outputs, the test suite catches it before anything ships.

The test script (`scripts/test_scan.py`) is the developer's primary tool. Run it after every sprint to verify the full pipeline end-to-end. It is not a unit test — it is a functional smoke test that exercises every layer of the system in sequence.

---

## 16. SUMMARY TABLE — WHAT BUILDS WHAT

| Module | Depends On | Produces |
|--------|-----------|---------|
| `ingestion/*` | External APIs, disk | `data/raw/*`, `data/processed/ingredients_master.csv` |
| `nlp/normaliser` | `ingredients_master.csv` | Canonical name for any input |
| `nlp/pubmed_extractor` | PubMed API, normaliser | `conflict_pairs_raw.csv` |
| `graph/builder` | All processed CSVs | `skingraph_v1.gpickle` |
| `graph/queries` | Graph pickle | Typed query results |
| `graph/confidence` | Graph edge + skin profile | `ConfidenceReport` |
| `scoring/*` | Graph queries | `ConflictResult` with scores |
| `models/train_lightgbm` | Feature builder + graph | `lgbm_risk_v1.pkl` |
| `models/predict` | `.pkl` file | `ModelScore` |
| `api/routes/scan` | All scoring + model | `ScanResponse` JSON |
| `explainer/*` | `ScanResponse` + Claude API | Plain language explanation |

---

*This document is version 1.0. Update it when architectural decisions change. Never let the code diverge from this spec silently.*