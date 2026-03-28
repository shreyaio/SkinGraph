from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal
from uuid import UUID
from skingraph.graph.schema import SkinProfile, ConfidenceReport

class BaseSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ProductInput(BaseSchema):
    name: str = Field(..., description="Product name (e.g., 'Minimalist Retinol 0.6%')")
    ingredients: Optional[List[str]] = Field(None, description="List of INCI ingredient names if known")
    routine_step: Optional[Literal["AM", "PM", "both"]] = Field(None, description="When the product is used")

class ScanRequest(BaseSchema):
    # Minimum 2 products for interaction analysis, maximum 5 for V1 routine limit
    products: List[ProductInput] = Field(..., min_length=2, max_length=5)
    skin_profile: Optional[SkinProfile] = None

class ConflictResult(BaseSchema):
    ingredient_a: str
    ingredient_b: str
    product_a: str
    product_b: str
    base_score: float
    skin_adjusted_score: float
    ml_score: float
    final_score: float
    severity: Literal["low", "medium", "high"]
    mechanism: str
    confidence: ConfidenceReport
    recommendation: str

class SynergyResult(BaseSchema):
    ingredient_a: str
    ingredient_b: str
    product_a: str
    product_b: str
    synergy_score: float
    mechanism: str
    recommendation: str

class OrderResult(BaseSchema):
    ingredient_a: str
    ingredient_b: str
    product_a: str
    product_b: str
    correct_order: str
    risk_if_reversed: str

class RoutineOutput(BaseSchema):
    am_routine: List[str] = Field(default_factory=list)
    pm_routine: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

class ScanResponse(BaseSchema):
    scan_id: UUID
    conflicts: List[ConflictResult] = Field(default_factory=list)
    synergies: List[SynergyResult] = Field(default_factory=list)
    order_issues: List[OrderResult] = Field(default_factory=list)
    routine_recommendation: RoutineOutput
    overall_risk_level: Literal["safe", "caution", "high_risk"]
    cumulative_burden_score: float

class ExplainRequest(BaseSchema):
    scan_result: ScanResponse
    skin_profile: SkinProfile

class ExplainResponse(BaseSchema):
    explanation: str
    word_count: int
