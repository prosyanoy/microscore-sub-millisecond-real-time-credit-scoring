"""HTTP load test against a running MicroScore API.

    locust -f benchmarks/locustfile.py --headless -u 32 -r 32 -t 20s --host http://127.0.0.1:8000
"""
import random

from locust import FastHttpUser, constant, task


def application() -> dict:
    return {
        "age": random.randint(21, 75),
        "annual_income": random.randint(20_000, 250_000),
        "loan_amount": random.randint(1_000, 40_000),
        "term_months": random.choice([12, 24, 36, 48, 60]),
        "interest_rate": round(random.uniform(4, 28), 2),
        "debt_to_income": round(random.uniform(0.02, 1.2), 3),
        "credit_utilization": round(random.uniform(0, 1.1), 3),
        "open_accounts": random.randint(0, 20),
        "delinquencies_2y": random.choice([0, 0, 0, 1, 2, 4]),
        "credit_history_months": random.randint(6, 400),
        "employment_years": round(random.uniform(0, 30), 1),
        "home_ownership": random.randint(0, 2),
        "purpose": random.randint(0, 4),
    }


class Scorer(FastHttpUser):
    wait_time = constant(0)

    @task(10)
    def score(self):
        self.client.post("/score", json=application(), name="/score")

    @task(1)
    def batch(self):
        self.client.post("/score/batch", json={"applications": [application() for _ in range(50)]},
                         name="/score/batch")
