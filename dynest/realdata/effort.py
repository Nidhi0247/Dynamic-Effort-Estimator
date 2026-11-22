"""Measured effort proxy from the status history ("active work hours").

TAWOS worklogs (Timespent) are sparse: ~10k of 458k issues, and only 2 of the 22 study projects (DM, MDL)
log time on more than a handful of modelling rows, so effort is measured from status intervals.  Validity check
(experiments/e0_proxy_validity.py, 15 projects with >= 50 logged issues): median Spearman rho with logged time
0.35, positive in 15/15 projects, at least as high as TAWOS' own In_Progress / Total_Effort minutes.

  1. an interval is *active* if its status name denotes work being done (ACTIVE rule below);
  2. its length is counted in business hours (weekdays, 8 h per weekday);
  3. the assignee's time is shared equally among all of their simultaneously active issues
     (work-in-progress adjustment), so parallel tickets are not double counted.

effort_h(issue) = sum over its active intervals of  integral( 8/24 * weekday(t) / WIP_assignee(t) dt )
This is a measurement definition (no fitted parameters); it is reported as a proxy in the paper.
"""
import re

import numpy as np
import pandas as pd

ACTIVE = re.compile(r"progress|in review|code review|peer review|review in progress|testing in progress|"
                    r"in testing|development|in dev|coding|implement|fixing|quality review", re.I)
NOT_ACTIVE = re.compile(r"waiting|needs|gathering|pending|ready for|blocked|backlog|to do|open", re.I)


def is_active(status):
    s = str(status)
    return bool(ACTIVE.search(s)) and not NOT_ACTIVE.search(s)


def weekday_hours(a, b):
    """Calendar hours between a and b that fall on Monday-Friday (vectorised, datetime64 inputs)."""
    a = np.asarray(a, "datetime64[s]")
    b = np.asarray(b, "datetime64[s]")
    da, db = a.astype("datetime64[D]"), b.astype("datetime64[D]")
    full = np.busday_count(da, db).astype(float) * 24.0
    fa = (a - da).astype(float) / 3600.0
    fb = (b - db).astype(float) / 3600.0
    wa, wb = np.is_busday(da), np.is_busday(db)
    return np.maximum(full - np.where(wa, fa, 0.0) + np.where(wb, fb, 0.0), 0.0)


def active_intervals(status_changes, issues):
    """status_changes: changelog rows (field 'status'); issues: frame with issue_id, Creation_Date,
    Resolution_Date, Assignee_ID.  Returns intervals (issue_id, start, end, status, assignee)."""
    ch = status_changes[["Issue_ID", "Creation_Date", "From_String", "To_String"]].sort_values(
        ["Issue_ID", "Creation_Date"])
    ch = ch.rename(columns={"Issue_ID": "issue_id", "Creation_Date": "start", "To_String": "status"})
    meta = issues.set_index("issue_id")
    ch["end"] = ch.groupby("issue_id")["start"].shift(-1)
    res = ch["issue_id"].map(meta["Resolution_Date"])
    ch["end"] = ch["end"].fillna(res)
    ch = ch.dropna(subset=["end"])
    ch = ch[ch["end"] > ch["start"]]
    ch = ch[ch["status"].map(is_active)]
    ch["assignee"] = ch["issue_id"].map(meta["Assignee_ID"])
    return ch[["issue_id", "start", "end", "status", "assignee"]]


def wip_adjusted_hours(intervals):
    """Per-interval business hours shared with the assignee's concurrently active intervals."""
    out = np.zeros(len(intervals))
    iv = intervals.reset_index(drop=True)
    s = iv["start"].values.astype("datetime64[s]")
    e = iv["end"].values.astype("datetime64[s]")
    plain = weekday_hours(s, e) * (8.0 / 24.0)
    noass = iv["assignee"].isna().values
    out[noass] = plain[noass]
    for _, idx in iv.groupby("assignee", dropna=True).indices.items():     # positions within iv
        ss, ee = s[idx], e[idx]
        T = np.unique(np.r_[ss, ee])
        cnt = np.zeros(len(T))
        np.add.at(cnt, np.searchsorted(T, ss), 1)
        np.add.at(cnt, np.searchsorted(T, ee), -1)
        c = np.cumsum(cnt)[:-1]                                   # active count on segment [T_k, T_k+1)
        bh = weekday_hours(T[:-1], T[1:]) * (8.0 / 24.0)
        P = np.r_[0.0, np.cumsum(np.where(c > 0, bh / np.maximum(c, 1), 0.0))]
        out[idx] = P[np.searchsorted(T, ee)] - P[np.searchsorted(T, ss)]
    return out


def effort_proxy(status_changes, issues):
    iv = active_intervals(status_changes, issues).reset_index(drop=True)
    iv = iv.assign(hours=wip_adjusted_hours(iv), plain_hours=weekday_hours(
        iv.start.values.astype("datetime64[s]"), iv.end.values.astype("datetime64[s]")) * 8 / 24)
    g = iv.groupby("issue_id")
    return pd.DataFrame(dict(effort_h=g.hours.sum(), active_bh=g.plain_hours.sum(), work_start=g.start.min(),
                             work_end=g.end.max(), n_active_intervals=g.size())), iv
