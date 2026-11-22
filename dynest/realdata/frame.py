"""Requirement-level modelling frame (S1 inputs + training labels), built strictly as-of.

Decision time t_i (when the estimate is made):
    t_i = min(estimation time, first sprint assignment, first active work), where estimation time is
    TAWOS' Estimation_Date (falls back to creation).
Features use only data stamped <= t_i (field values reconstructed from the changelog):
    title/description text, story points, priority, type, #components, #links, description length,
    acceptance-criteria lines, comments so far, reporter's prior issues, assignee familiarity/tenure.
Labels (training use only) come from the issue's later history:
    effort_h          measured active work hours (dynest.realdata.effort)
    clarity_label     -log1p(#summary/description/SP edits after t_i + #question comments after t_i)
    integration_label log1p(#links at resolution + #components at resolution)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .effort import effort_proxy
from .store import Store, count_after, count_before, field_as_of

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"
DONE_RESOLUTIONS = {"Fixed", "Done", "Complete", "Resolved", "Implemented", "Delivered", "Answered"}
_AC = re.compile(r"^\s*([*#-]|\d+[.)]|given |when |then |ac\d*|acceptance)", re.I | re.M)
_LINK_REL = [
    (re.compile(r"This issue (blocks|is a dependency of|is depended on by|is required by|has to be done before) (\S+)", re.I), "out"),
    (re.compile(r"This issue (is blocked by|depends on|requires|has to be done after) (\S+)", re.I), "in"),
]


def _clean(text):
    t = re.sub(r"\{code[^}]*\}.*?\{code\}|\{noformat\}.*?\{noformat\}", " ", str(text or ""), flags=re.S)
    t = re.sub(r"\[\~?[^\]]*\]|https?://\S+|[{}*_|#!]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def dependency_events(link_changes):
    """Precedence edges with timestamps from Link changelog text: (pred_key, succ_key, ts, added?)."""
    rows = []
    for r in link_changes.itertuples():
        for txt, added in ((r.To_String, True), (r.From_String, False)):
            if not isinstance(txt, str):
                continue
            for rx, direction in _LINK_REL:
                m = rx.search(txt)
                if m:
                    rows.append((r.Issue_ID, m.group(2).rstrip(".,"), direction, r.Creation_Date, added))
    return pd.DataFrame(rows, columns=["issue_id", "other_key", "direction", "ts", "added"])


class FrameContext:
    """Loads a set of projects once; computes as-of features for any (issue, time) pairs cheaply."""

    def __init__(self, project_ids, store=None, log=print):
        st = store or Store()
        self.st = st
        iss = st.issues
        iss = iss[iss.project_id.isin(project_ids)].copy()
        self.ids = set(iss.issue_id)
        log(f"  issues in scope: {len(iss):,}")
        cl = st.changelog(fields=["status", "Sprint", "Story Points", "summary", "description", "priority",
                                  "issuetype", "Component", "Link", "assignee"], issue_ids=self.ids)
        self.byf = {f: g for f, g in cl.groupby("Field")}
        self.empty = cl.iloc[:0]
        log(f"  changelog rows: {len(cl):,}")
        eff, self.intervals = effort_proxy(self.byf.get("status", self.empty), iss)
        iss = iss.merge(eff, left_on="issue_id", right_index=True, how="left")
        spr = self.byf.get("Sprint", self.empty)
        iss["first_sprint_at"] = iss.issue_id.map(spr[spr.To_Value.notna()].groupby("Issue_ID").Creation_Date.min())
        iss = iss.sort_values("Creation_Date")
        iss["reporter_prior"] = iss.groupby(["project_id", "Reporter_ID"]).cumcount()
        self.iss = iss
        self.fin = iss.set_index("issue_id")
        self.comp_final = st.components[st.components.Issue_ID.isin(self.ids)].groupby("Issue_ID").size()
        com = st.comments(self.ids)
        self.com = com.assign(q=com.Comment_Text.fillna("").str.contains(r"\?"))
        res = iss[["project_id", "Assignee_ID", "Resolution_Date"]].dropna().sort_values("Resolution_Date")
        self.res_by = {k: v.Resolution_Date.values for k, v in res.groupby(["project_id", "Assignee_ID"])}
        self.first_seen = iss.groupby("Assignee_ID").Creation_Date.min()

    def _ch(self, f, ids):
        g = self.byf.get(f, self.empty)
        return g[g.Issue_ID.isin(ids)]

    def features_as_of(self, T):
        """T: Series issue_id -> decision time.  Returns one row per issue with as-of features only."""
        T = T[~T.index.duplicated()]
        ids = set(T.index)
        fin = self.fin.loc[list(T.index)]
        out = pd.DataFrame(index=T.index)
        out["t_decision"] = T.values
        sp_final = fin.Story_Point.astype(object).map(lambda v: None if pd.isna(v) else str(v))
        out["sp"] = pd.to_numeric(field_as_of(sp_final, self._ch("Story Points", ids), T), errors="coerce")
        for f, col in (("summary", "Title"), ("description", "Description_Text"), ("priority", "Priority"),
                       ("issuetype", "Type")):
            out[f"{f}_t"] = field_as_of(fin[col].astype(object), self._ch(f, ids), T)
        out["text"] = (out.summary_t.fillna("").map(_clean) + ". "
                       + out.description_t.fillna("").map(_clean)).str.slice(0, 4000)
        out["desc_words"] = out.description_t.fillna("").map(lambda s: len(str(s).split()))
        out["ac_lines"] = out.description_t.fillna("").map(lambda s: len(_AC.findall(str(s))))
        cc = self._ch("Component", ids)
        out["n_components_t"] = (out.index.to_series().map(self.comp_final).fillna(0)
                                 - out.index.to_series().map(count_after(cc[cc.To_Value.notna()], T)).fillna(0)
                                 + out.index.to_series().map(count_after(cc[cc.From_Value.notna()], T)).fillna(0)).clip(lower=0)
        lk = self._ch("Link", ids)
        out["n_links_t"] = (out.index.to_series().map(count_before(lk[lk.To_Value.notna()], T)).fillna(0)
                            - out.index.to_series().map(count_before(lk[lk.From_Value.notna()], T)).fillna(0)).clip(lower=0)
        com = self.com[self.com.Issue_ID.isin(ids)]
        out["comments_t"] = out.index.to_series().map(count_before(com, T)).fillna(0)
        ass_final = fin.Assignee_ID.astype(object).map(lambda v: None if pd.isna(v) else str(int(v)))
        out["assignee_t"] = pd.to_numeric(field_as_of(ass_final, self._ch("assignee", ids), T,
                                                      from_col="From_Value"), errors="coerce")
        out["project_id"] = fin.project_id.values
        out["created_at"] = fin.Creation_Date.values
        out["reporter_prior"] = fin.reporter_prior.values
        fam, ten = [], []
        for pid, a, t in zip(out.project_id, out.assignee_t, out.t_decision):
            arr = self.res_by.get((pid, a))
            fam.append(0 if arr is None else int(np.searchsorted(arr, np.datetime64(t), side="left")))
            fs = self.first_seen.get(a)
            ten.append(0.0 if fs is None or pd.isna(fs) else max((t - fs).days, 0))
        out["familiarity"], out["tenure_days"] = fam, ten
        out["age_days_t"] = (out.t_decision - out.created_at).dt.total_seconds() / 86400
        out.index.name = "issue_id"
        return out.reset_index()


_CTX = {}


def context(project_ids, store=None, log=print):
    key = tuple(sorted(project_ids))
    if key not in _CTX:
        _CTX[key] = FrameContext(project_ids, store, log)
    return _CTX[key]


def build_frame(project_ids, store=None, use_cache=True, log=print):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = "frame_" + "_".join(map(str, sorted(project_ids))) + ".parquet"
    if use_cache and (CACHE / key).exists():
        return pd.read_parquet(CACHE / key)
    ctx = context(project_ids, store, log)
    iss = ctx.iss
    # The estimate is used when work begins: t = one second before the first active-status interval
    # (no work has been done yet, so nothing about the work itself can leak).  TAWOS' Estimation_Date
    # is a creation-time artefact (creation - 1 h) and is not used.  Issues never worked on fall back
    # to their first sprint assignment, then creation.
    t = (iss.work_start - pd.Timedelta(seconds=1)).fillna(iss.first_sprint_at).fillna(iss.Creation_Date)
    T = pd.Series(t.clip(lower=iss.Creation_Date).values, index=iss.issue_id.values)
    feats = ctx.features_as_of(T)
    meta = iss[["issue_id", "Issue_Key", "Resolution_Date", "Resolution", "work_start", "work_end",
                "first_sprint_at", "Sprint_ID", "Assignee_ID", "Reporter_ID", "effort_h", "active_bh",
                "n_active_intervals", "Story_Point", "In_Progress_Minutes"]]
    out = feats.merge(meta, on="issue_id", how="left")
    ass = ctx.byf.get("assignee", ctx.empty)
    out["n_assignees"] = out.issue_id.map(ass[ass.To_Value.notna()].groupby("Issue_ID").To_Value.nunique()).fillna(0)

    # ---------------------------------------------------------------- labels (after t, up to resolution)
    edits = pd.concat([ctx.byf.get(f, ctx.empty) for f in ("summary", "description", "Story Points")])
    until = out.set_index("issue_id").Resolution_Date
    Tq = out.set_index("issue_id").t_decision
    n_edits = count_after(edits, Tq, until)
    n_q = count_after(ctx.com[ctx.com.q], Tq, until)
    out["clarity_label"] = -np.log1p(out.issue_id.map(n_edits).fillna(0) + out.issue_id.map(n_q).fillna(0))
    R = until.fillna(pd.Timestamp.max)
    lk = ctx.byf.get("Link", ctx.empty)
    cc = ctx.byf.get("Component", ctx.empty)
    links_res = (count_before(lk[lk.To_Value.notna()], R).reindex(out.issue_id).fillna(0).values
                 - count_before(lk[lk.From_Value.notna()], R).reindex(out.issue_id).fillna(0).values).clip(min=0)
    comp_res = (out.issue_id.map(ctx.comp_final).fillna(0).values
                - out.issue_id.map(count_after(cc[cc.To_Value.notna()], R)).fillna(0).values
                + out.issue_id.map(count_after(cc[cc.From_Value.notna()], R)).fillna(0).values).clip(min=0)
    out["integration_label"] = np.log1p(links_res + comp_res)

    out = out.rename(columns={"Issue_Key": "issue_key", "Resolution_Date": "resolved_at", "Sprint_ID": "sprint_id"})
    out["done"] = out.Resolution.isin(DONE_RESOLUTIONS)
    out = out.sort_values("t_decision").reset_index(drop=True)
    out.to_parquet(CACHE / key)
    ctx.intervals.to_parquet(CACHE / key.replace("frame_", "intervals_"))
    return out


def modelling_rows(frame, min_hours=0.25, max_sp=100):
    """Rows usable for supervised estimation: done, SP known at decision time, measured effort."""
    m = (frame.done & frame.sp.between(0.5, max_sp) & (frame.effort_h >= min_hours) & frame.resolved_at.notna()
         & (frame.t_decision < frame.work_start))
    return frame[m].copy()
