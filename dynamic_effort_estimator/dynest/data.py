from pathlib import Path
import numpy as np
import pandas as pd

CLUSTERS = ["UI-heavy", "Backend-intensive", "Data-processing"]
FEATURES = ["ui", "backend", "data", "complexity", "integration"]  
ATTRS = FEATURES + ["clarity", "experience", "size_points"]
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "historical_requirements.csv"
_TYPE_BIAS = np.array([-0.3, 0.6, 0.4])

def true_complexity(complexity, integration, kind, rng=None):
    c = 0.5 * complexity + 0.3 * integration + 0.02 * complexity * integration + _TYPE_BIAS[kind]
    if rng is not None:
        c = c + rng.normal(0, 0.3, size=np.shape(c))
    return np.clip(c, 0, 10)


def intensity(rating, clarity, experience):
    return (0.1 + 0.16 * rating) * (1 + 0.04 * (10 - clarity)) * (1.3 - 0.04 * experience)


def make_history(n=900, seed=42):
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 3, n)
    prof = rng.uniform(0, 4, (n, 3))
    prof[np.arange(n), kind] = rng.uniform(6, 10, n)
    comp = rng.uniform(1, 10, n)
    integ = np.clip(rng.uniform(1, 10, n) + (kind == 1), 1, 10)
    clarity = rng.uniform(1, 10, n)
    exp = rng.uniform(1, 10, n)
    size = rng.choice([1, 2, 3, 5, 8, 13], n)
    rating = true_complexity(comp, integ, kind, rng)
    actual = size * intensity(rating, clarity, exp) * rng.lognormal(0, 0.12, n)
    df = pd.DataFrame(dict(ui=prof[:, 0], backend=prof[:, 1], data=prof[:, 2], complexity=comp,
                           integration=integ, clarity=clarity, experience=exp, size_points=size,
                           assessed_complexity=rating, actual_effort=actual)).round(3)
    return df


def load_history():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    df = make_history()
    DATA_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    return df


def demo_project():
    rows = [
        ("R01", "User login & SSO",            3, 8, 2, 5, 6, 8, 7, 5, 3),
        ("R02", "Responsive landing pages",    9, 2, 1, 3, 2, 9, 8, 5, 2),
        ("R03", "Payment gateway integration", 2, 9, 3, 8, 9, 6, 5, 8, 3),
        ("R04", "Admin analytics dashboard",   8, 3, 7, 6, 5, 5, 6, 8, 2),
        ("R05", "Product catalogue API",       2, 9, 4, 5, 4, 8, 8, 5, 3),
        ("R06", "Recommendation engine",       1, 5, 9, 9, 7, 3, 4, 13, 1),
        ("R07", "Order tracking screens",      9, 3, 2, 4, 4, 8, 7, 3, 2),
        ("R08", "ETL pipeline for sales data", 1, 4, 9, 7, 8, 6, 6, 8, 2),
        ("R09", "Notification service",        3, 8, 3, 5, 6, 7, 7, 5, 2),
        ("R10", "Customer reports export",     6, 3, 8, 4, 3, 9, 8, 3, 1),
        ("R11", "Role-based access control",   3, 9, 2, 6, 7, 7, 6, 5, 3),
        ("R12", "Mobile-friendly checkout UI", 9, 2, 1, 5, 5, 6, 6, 5, 3),
    ]
    cols = ["id", "title", "ui", "backend", "data", "complexity", "integration",
            "clarity", "experience", "size_points", "priority"]
    df = pd.DataFrame(rows, columns=cols)
    df["active"] = True
    return df


def demo_team():
    return pd.DataFrame([
        dict(name="Asha",  specialty="UI-heavy",         speed=1.0),
        dict(name="Ravi",  specialty="Backend-intensive", speed=1.0),
        dict(name="Meera", specialty="Data-processing",   speed=1.0),
        dict(name="Kabir", specialty="Generalist",        speed=0.95),
    ])
