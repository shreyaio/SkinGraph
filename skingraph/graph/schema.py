from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

class RelationType(str, Enum):
    CONFLICT = "CONFLICT"
    SYNERGY = "SYNERGY"
    ORDER_SENSITIVE = "ORDER_SENSITIVE"
    NEUTRAL = "NEUTRAL"
    UNCLEAR = "UNCLEAR"

@dataclass
class NodeData:
    canonical_inci: str
    pubchem_cid: Optional[int] = None
    function_labels: List[str] = field(default_factory=list)
    irritancy_level: Optional[str] = None
    acne_risk: Optional[str] = None
    ewg_score: Optional[int] = None
    ph_optimal: Optional[Tuple[float, float]] = None
    skin_type_caution: List[str] = field(default_factory=list)
    fitzpatrick_notes: Optional[str] = None

@dataclass
class EdgeData:
    relation_type: RelationType
    base_score: float
    source_count: int
    agreement_ratio: float
    primary_mechanism: str
    fitzpatrick_iv_v_flag: bool
    pih_risk: bool
    confidence_tier: str
    sources: List[str] = field(default_factory=list)
    safe_at_concentrations: Optional[str] = None
    order_note: Optional[str] = None

@dataclass
class SkinProfile:
    skin_type: str
    fitzpatrick: str
    sensitivity: str
    concerns: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)

@dataclass
class ConfidenceReport:
    score: float
    tier: str
    source_count: int
    uncertainty_note: Optional[str] = None

@dataclass
class RiskMatrix:
    edges: List[EdgeData]
    cumulative_burden_score: float
