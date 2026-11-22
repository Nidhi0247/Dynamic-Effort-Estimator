"""Canonical event store over the converted TAWOS parquet files, plus as-of reconstruction.

The leakage rule of the whole project lives here: `field_as_of(field, t)` returns, per issue, the
value the field had at time t (by undoing every changelog entry made after t), and `assert_as_of`
checks that no derived timestamp exceeds the decision time.
"""
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "tawos"


class Store:
    def __init__(self, root=ROOT):
        self.root = Path(root)

    def _read(self, name, columns=None):
        return pd.read_parquet(self.root / f"{name}.parquet", columns=columns)

    # ------------------------------------------------------------------ base tables
    @cached_property
    def projects(self):
        return self._read("Project")

    @cached_property
    def issues(self):
        cols = ["ID", "Issue_Key", "Title", "Description_Text", "Type", "Priority", "Status", "Resolution",
                "Creation_Date", "Estimation_Date", "Resolution_Date", "Story_Point", "Timespent",
                "In_Progress_Minutes", "Total_Effort_Minutes", "Resolution_Time_Minutes", "Creator_ID",
                "Reporter_ID", "Assignee_ID", "Project_ID", "Sprint_ID"]
        df = self._read("Issue", cols)
        return df.rename(columns={"ID": "issue_id", "Project_ID": "project_id"})

    @cached_property
    def sprints(self):
        return self._read("Sprint").rename(columns={"ID": "sprint_id", "Project_ID": "project_id"})

    @cached_property
    def links(self):
        return self._read("Issue_Link").rename(columns={"Issue_ID": "src", "Target_Issue_ID": "dst"})

    @cached_property
    def components(self):
        ic = self._read("Issue_Component")
        c = self._read("Component", ["ID", "Name", "Project_ID"]).rename(columns={"ID": "Component_ID"})
        return ic.merge(c, on="Component_ID", how="left")

    def changelog(self, fields=None, issue_ids=None):
        df = self._read("Change_Log", ["Field", "From_Value", "To_Value", "From_String", "To_String",
                                        "Creation_Date", "Author_ID", "Issue_ID"])
        if fields is not None:
            df = df[df["Field"].isin(fields)]
        if issue_ids is not None:
            df = df[df["Issue_ID"].isin(issue_ids)]
        return df.dropna(subset=["Issue_ID"]).astype({"Issue_ID": "int64"}).sort_values("Creation_Date")

    def comments(self, issue_ids=None):
        df = self._read("Comment", ["Comment_Text", "Creation_Date", "Author_ID", "Issue_ID"])
        if issue_ids is not None:
            df = df[df["Issue_ID"].isin(issue_ids)]
        return df


def field_as_of(final, changes, t, value_col="To_String", from_col="From_String"):
    """Value of a field at time t for each issue.

    final   : Series indexed by issue_id with the final (current) value
    changes : changelog rows for that field (Issue_ID, Creation_Date, From_*, To_*)
    t       : Series indexed by issue_id with the decision time
    The earliest change made after t tells us the value just before it (its From value); with no
    change after t the final value holds.
    """
    ch = changes[["Issue_ID", "Creation_Date", from_col]].copy()
    ch["t"] = ch["Issue_ID"].map(t)
    after = ch[ch["Creation_Date"] > ch["t"]].sort_values("Creation_Date")
    first_after = after.groupby("Issue_ID")[from_col].first()
    out = final.copy()
    idx = first_after.index.intersection(out.index)
    out.loc[idx] = first_after.loc[idx].values
    return out


def count_before(changes, t, by="Issue_ID"):
    ch = changes[[by, "Creation_Date"]].copy()
    ch["t"] = ch[by].map(t)
    return ch[ch["Creation_Date"] <= ch["t"]].groupby(by).size()


def count_after(changes, t, until=None, by="Issue_ID"):
    ch = changes[[by, "Creation_Date"]].copy()
    ch["t"] = ch[by].map(t)
    m = ch["Creation_Date"] > ch["t"]
    if until is not None:
        m &= ch["Creation_Date"] <= ch[by].map(until)
    return ch[m].groupby(by).size()


def assert_as_of(times, decision_times, what="feature"):
    bad = (pd.to_datetime(times) > pd.to_datetime(decision_times)).sum()
    if bad:
        raise AssertionError(f"leakage: {bad} {what} timestamps are after the decision time")
