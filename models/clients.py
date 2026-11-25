"""Client profiles: what we know about a client after the survey.

IndividualProfile and InstitutionalProfile are the survey outputs. models/spec.py turns
either one into a PortfolioSpec, which is all the optimizer needs.

ClientProfile is the original simple profile, kept so existing code and tests still work.
"""
from dataclasses import dataclass, field

from models.goals import Goal


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


@dataclass
class IndividualProfile:
    """A person or household.

    risk_tolerance: 1 (very cautious) to 10 (very aggressive), from the survey. This is the
        client's willingness to take risk; their ability to take risk comes from the goals.
    goals: what the money is for (see models/goals.py). Ages are the client's age.
    annual_savings: how much they add per year, in today's dollars, until savings_until_age.
    horizon_years: only used if there are no goals.
    """

    name: str
    age: int
    wealth: float
    risk_tolerance: int
    goals: list[Goal] = field(default_factory=list)
    annual_savings: float = 0.0
    savings_until_age: int | None = None
    horizon_years: float | None = None
    min_esg_share: float = 0.0
    excluded_tickers: frozenset[str] = frozenset()
    max_weight: float = 0.25


@dataclass
class InstitutionalProfile:
    """A pension plan or an endowment/foundation.

    kind: "pension" (has promised benefit payments) or "endowment" (spends a share of assets
        each year, forever).
    risk_tolerance: 1 to 10, as agreed by the investment committee.
    liabilities: pension only. Projected benefit payments as Goals, with start/end in years
        from now (current age is 0).
    spending_rate: endowment only. Share of assets paid out each year (typically 0.04-0.05).
    policy_ranges: asset-class ranges from the investment policy statement, e.g.
        {"equity": (0.3, 0.6)}. These override our own rules for those asset classes.
    liability_proxy: the fund whose returns best mimic the pension's liabilities.
    """

    name: str
    kind: str
    assets: float
    risk_tolerance: int
    liabilities: list[Goal] = field(default_factory=list)
    spending_rate: float = 0.0
    policy_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    liability_proxy: str = "TLT"
    min_esg_share: float = 0.0
    excluded_tickers: frozenset[str] = frozenset()
    max_weight: float = 0.25


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

EXAMPLE_INDIVIDUALS = [
    IndividualProfile(
        "Young professional",
        age=25,
        wealth=60_000,
        risk_tolerance=8,
        annual_savings=15_000,
        savings_until_age=60,
        goals=[
            Goal("House down payment", 80_000, start=30),
            Goal("Retirement income", 4_000, start=60, end=90, payments_per_year=12),
            Goal("Inheritance for children", 250_000, start=90, essential=False),
        ],
    ),
    IndividualProfile(
        "Near-retiree",
        age=60,
        wealth=900_000,
        risk_tolerance=4,
        goals=[Goal("Retirement income", 4_500, start=63, end=90, payments_per_year=12)],
    ),
    IndividualProfile(
        "ESG-focused investor",
        age=40,
        wealth=300_000,
        risk_tolerance=6,
        annual_savings=20_000,
        savings_until_age=65,
        min_esg_share=0.5,
        excluded_tickers=frozenset({"GLD"}),
        goals=[Goal("Retirement income", 5_000, start=65, end=90, payments_per_year=12)],
    ),
]

EXAMPLE_INSTITUTIONS = [
    InstitutionalProfile(
        "Regional pension plan",
        kind="pension",
        assets=500_000_000,
        risk_tolerance=5,
        liabilities=[Goal("Benefit payments", 30_000_000, start=1, end=30)],
    ),
    InstitutionalProfile(
        "University endowment",
        kind="endowment",
        assets=1_000_000_000,
        risk_tolerance=7,
        spending_rate=0.045,
        policy_ranges={"real_assets": (0.05, 0.20)},
    ),
]
