"""Client profiles: turn a client's characteristics into inputs for the optimizer."""
from dataclasses import dataclass


@dataclass
class ClientProfile:
    name: str
    wealth: float  # dollars
    risk_tolerance: int  # 1 (very cautious) to 10 (very aggressive)
    horizon_years: int
    min_esg_share: float = 0.0  # minimum share of the portfolio in ESG-labelled funds (0 to 1)
    max_weight: float = 0.4  # cap on any single fund

    def risk_aversion(self) -> float:
        """Placeholder mapping from risk tolerance to a mean-variance risk-aversion parameter.

        Tolerance 10 gives 1 and tolerance 1 gives 10. Replace once the team agrees on a mapping.
        """
        return 11 - self.risk_tolerance


EXAMPLE_CLIENTS = [
    ClientProfile("Young growth investor", wealth=50_000, risk_tolerance=9, horizon_years=35),
    ClientProfile("Near-retiree", wealth=800_000, risk_tolerance=3, horizon_years=5),
    ClientProfile(
        "ESG-focused investor",
        wealth=200_000,
        risk_tolerance=6,
        horizon_years=20,
        min_esg_share=0.5,
    ),
]
