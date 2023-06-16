"""Feature schema shared by training, serving and benchmarks."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

FEATURES: list[str] = [
    "age",
    "annual_income",
    "loan_amount",
    "term_months",
    "interest_rate",
    "debt_to_income",
    "credit_utilization",
    "open_accounts",
    "delinquencies_2y",
    "credit_history_months",
    "employment_years",
    "home_ownership",  # 0 rent, 1 mortgage, 2 own
    "purpose",  # 0 debt consolidation, 1 home, 2 car, 3 education, 4 other
]


class Application(BaseModel):
    """A single loan application. `employment_years` may be omitted (missing)."""

    model_config = {"extra": "forbid"}

    age: float = Field(ge=18, le=100)
    annual_income: float = Field(ge=0, le=10_000_000)
    loan_amount: float = Field(gt=0, le=5_000_000)
    term_months: int = Field(ge=6, le=120)
    interest_rate: float = Field(ge=0, le=60, description="percent")
    debt_to_income: float = Field(ge=0, le=5)
    credit_utilization: float = Field(ge=0, le=1.5)
    open_accounts: int = Field(ge=0, le=100)
    delinquencies_2y: int = Field(ge=0, le=50)
    credit_history_months: int = Field(ge=0, le=1200)
    employment_years: Optional[float] = Field(default=None, ge=0, le=70)
    home_ownership: int = Field(ge=0, le=2)
    purpose: int = Field(ge=0, le=4)

    def to_row(self) -> list[float]:
        row = []
        for name in FEATURES:
            v = getattr(self, name)
            row.append(math.nan if v is None else float(v))
        return row


def applications_to_matrix(apps: list[Application]) -> np.ndarray:
    return np.asarray([a.to_row() for a in apps], dtype=np.float64).reshape(len(apps), len(FEATURES))


def row_to_dict(row) -> dict:
    return {n: (None if (isinstance(v, float) and math.isnan(v)) else v) for n, v in zip(FEATURES, row)}
