"""Classic project-level effort datasets (PROMISE / Zenodo public copies) with a uniform schema.

Each loader returns a ClassicDataset:
    X      DataFrame of predictors known at estimation time (post-hoc columns such as actual duration dropped)
    y      actual effort (dataset's own unit: hours or person-months)
    size   functional size (FP) or code size (KLOC) column name inside X
    kind   "FP" or "LOC"
    cats   categorical predictors (one-hot encoded inside each model on the training fold only)
    em     COCOMO-81 effort-multiplier columns (numeric multipliers), if the dataset has them
    mode   COCOMO development mode per project, if available
    expert human estimate made before the project (Kitchenham "First.estimate"), reference only

Download the raw files to data/classic_raw (see experiments/README or e8_classic.py):
    albrecht_full.arff kemerer_full.arff (Zenodo 268467 / 268464), the rest from the DASE book repository.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .arff import read_arff

RAW = Path(__file__).resolve().parents[2] / "data" / "classic_raw"
EM_COLS = ["rely", "data", "cplx", "time", "stor", "virt", "turn", "acap", "aexp", "pcap", "vexp", "lexp",
           "modp", "tool", "sced"]


@dataclass
class ClassicDataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    size: str
    kind: str
    unit: str
    cats: list = field(default_factory=list)
    em: list = field(default_factory=list)
    mode: np.ndarray = None
    expert: np.ndarray = None

    @property
    def n(self):
        return len(self.y)


def _finish(name, df, target, size, kind, unit, drop=(), cats=(), em=(), mode=None, expert=None):
    df = df.drop(columns=[c for c in drop if c in df]).copy()
    keep = df[target].notna() & (df[target] > 0) & df[size].notna() & (df[size] > 0)
    df = df[keep].reset_index(drop=True)
    y = df.pop(target).astype(float).values
    for c in cats:
        df[c] = df[c].astype(str)
    num = [c for c in df if c not in cats]
    df[num] = df[num].astype(float)
    df = df.dropna().reset_index(drop=True) if df.isna().any().any() else df
    if len(df) != len(y):                                      # rows with missing predictors removed
        raise ValueError(f"{name}: drop rows with missing predictors before calling _finish")
    return ClassicDataset(name, df, y, size, kind, unit, list(cats), list(em),
                          None if mode is None else np.asarray(mode)[keep.values],
                          None if expert is None else np.asarray(expert, float)[keep.values])


def _arff(f, block=0):
    return read_arff(RAW / f)[block]


def albrecht():
    return _finish("Albrecht", _arff("albrecht_full.arff"), "Effort", "AdjFP", "FP", "1000 h")


def kemerer():
    return _finish("Kemerer", _arff("kemerer_full.arff"), "EffortMM", "AdjFP", "FP", "PM",
                   drop=("ID", "Duration"), cats=("Language", "Hardware"))


def desharnais():
    df = _arff("desharnais.arff").dropna().reset_index(drop=True)
    return _finish("Desharnais", df, "Effort", "PointsAdjust", "FP", "h", cats=("Language",))


def china():
    return _finish("China", _arff("china.arff"), "Effort", "AFP", "FP", "h", drop=("Duration",),
                   cats=("Resource", "DevType"))


def maxwell():
    return _finish("Maxwell", _arff("maxwell.arff"), "effort_D", "size_D", "FP", "h", drop=("duration_D",),
                   cats=("app", "har", "dba", "ifc", "source", "telonuse"))


def kitchenham():
    df = _arff("kitchenham.arff")
    expert = df["First.estimate"].values
    return _finish("Kitchenham", df, "Actual.effort", "Adjusted.function.points", "FP", "h",
                   drop=("Actual.duration", "First.estimate"), expert=expert)


def miyazaki94():
    return _finish("Miyazaki94", _arff("miyazaki94.arff"), "MM", "KLOC", "LOC", "PM", drop=("ID",))


def _cocomo81_blocks():
    num, let = [b[b.project_id != 0].reset_index(drop=True) for b in read_arff(RAW / "coc81_1_1.arff")]
    return num, let


def em_table():
    """COCOMO-81 effort-multiplier table recovered from the paired numeric/ordinal encodings of COCOMO81:
    for every driver and rating, the modal numeric multiplier (it equals Boehm 1981, Table 8-2)."""
    num, let = _cocomo81_blocks()
    return {c: num.groupby(let[c])[c].agg(lambda s: s.mode().iloc[0]).to_dict() for c in EM_COLS}


def cocomo81():
    num, _ = _cocomo81_blocks()
    mode = num["dev_mode"].values
    df = num.drop(columns=["project_id"])
    return _finish("COCOMO81", df, "actual", "loc", "LOC", "PM", cats=("dev_mode",), em=EM_COLS, mode=mode)


def nasa93():
    df = _arff("nasa93.arff")
    tab = em_table()
    missing = []
    for c in EM_COLS:
        v = df[c].map(tab[c])
        missing += [(c, r) for r in df.loc[v.isna(), c].unique()]
        df[c] = v
    if missing:
        raise ValueError(f"NASA93 ratings without a COCOMO-81 multiplier: {missing}")
    mode = df["mode"].values
    df = df.drop(columns=["recordnumber", "projectname"])
    return _finish("NASA93", df, "act_effort", "equivphyskloc", "LOC", "PM", cats=("cat2", "forg", "center", "mode"),
                   em=EM_COLS, mode=mode)


LOADERS = dict(Albrecht=albrecht, Kemerer=kemerer, Desharnais=desharnais, China=china, Maxwell=maxwell,
               Kitchenham=kitchenham, Miyazaki94=miyazaki94, COCOMO81=cocomo81, NASA93=nasa93)


def load_all(names=None):
    return {k: f() for k, f in LOADERS.items() if names is None or k in names}
