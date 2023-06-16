"""Synthetic loan-application data with a non-linear default process."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FEATURES


def make_dataset(n: int = 60_000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X, y) where y=1 means the borrower defaulted. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 76, n).astype(float)
    income = np.clip(rng.lognormal(11.0, 0.55, n), 12_000, 600_000)
    loan_amount = np.clip(income * rng.uniform(0.04, 0.9, n), 500, 200_000)
    term = rng.choice([12, 24, 36, 48, 60, 72], n, p=[0.05, 0.15, 0.4, 0.15, 0.2, 0.05]).astype(float)
    dti = np.clip(rng.beta(2.2, 5.0, n) * 1.2 + 0.25 * loan_amount / income, 0, 3)
    util = np.clip(rng.beta(2.0, 2.6, n) + rng.normal(0, 0.05, n), 0, 1.4)
    open_acc = rng.poisson(8, n).astype(float)
    delinq = rng.poisson(np.clip(0.2 + 1.2 * util * dti, 0.05, 3), n).astype(float)
    history = np.clip(rng.normal(40 + 4.0 * (age - 21), 40, n), 6, 700).round()
    employment = np.clip(rng.gamma(2.0, 3.5, n) * (age - 18) / 40, 0, 50)
    home = rng.choice([0, 1, 2], n, p=[0.42, 0.40, 0.18]).astype(float)
    purpose = rng.choice([0, 1, 2, 3, 4], n, p=[0.45, 0.15, 0.15, 0.1, 0.15]).astype(float)

    base_rate = 5.5 + 9.0 * util * 0.5 + 2.5 * np.minimum(delinq, 4) + rng.normal(0, 1.2, n)
    rate = np.clip(base_rate, 2.0, 35.0)

    logit = (
        -3.6
        + 2.6 * np.maximum(dti - 0.45, 0) ** 1.2 * 3
        + 1.5 * util**2
        + 0.45 * np.minimum(delinq, 6)
        + 0.045 * (rate - 10)
        - 0.012 * np.minimum(history, 240) / 6
        - 0.07 * np.minimum(employment, 15)
        + 0.35 * (term >= 60)
        + 0.5 * (loan_amount / income > 0.5)
        - 0.25 * (home == 2)
        + 0.2 * (purpose == 3)
        + 0.5 * (util > 0.8) * (dti > 0.6)  # interaction a linear model would miss
        - 0.012 * np.abs(age - 45)
        + rng.normal(0, 0.35, n)
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)

    employment = np.where(rng.random(n) < 0.08, np.nan, employment)  # missing values
    X = pd.DataFrame(
        {
            "age": age,
            "annual_income": income.round(0),
            "loan_amount": loan_amount.round(0),
            "term_months": term,
            "interest_rate": rate.round(2),
            "debt_to_income": dti.round(4),
            "credit_utilization": util.round(4),
            "open_accounts": open_acc,
            "delinquencies_2y": delinq,
            "credit_history_months": history,
            "employment_years": employment.round(2),
            "home_ownership": home,
            "purpose": purpose,
        }
    )[FEATURES]
    return X, y
