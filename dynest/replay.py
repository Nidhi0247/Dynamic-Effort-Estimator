"""Real-history replay (S7/S8): turn a project window into an EpisodeData.

Episode = project p, start T0 (a real sprint start inside the evaluation window), n sprints of the
project's median sprint length.
  scope      issues that entered a sprint by the episode end and were resolved after T0
  day-0 tasks  entered a sprint before T0; ADD events at their first sprint assignment inside the window
  DROP       resolved with a not-done resolution inside the window (work done on them is wasted)
  MODIFY     story-point changes inside the window -> re-estimate as of the change
  truth      remaining measured effort after T0 (WIP-adjusted active hours), converted to reference
             hours with the in-sample efficiency of the developer who really did the work;
             capacity = each developer's real active hours per sprint
  planner    estimator predictions as of each event, efficiency + capacity learnt on training data only
  due sprint the end of the real sprint the issue was committed to when it entered the plan
"""
import numpy as np
import pandas as pd

from .execution import EfficiencyModel, CapacityForecaster
from .realdata.effort import weekday_hours
from .realdata.frame import dependency_events, DONE_RESOLUTIONS
from .scheduling import ScheduleInstance
from .sim import EpisodeData, Task, execute


class ReplayBuilder:
    def __init__(self, ctx, frame):
        self.ctx, self.frame = ctx, frame.set_index("issue_id")
        self.iv = ctx.intervals.assign(hours=ctx.intervals.hours.astype(float))
        st = ctx.st
        sp = st.sprints.copy()
        # a sprint *series* = one team's board (sprint name without its number/date), e.g. "db_s", "arch"
        sp["series"] = sp.Name.fillna("").str.replace(r"[\d].*$", "", regex=True).str.strip(" -_#:").str.lower()
        self.sprints = sp
        self.key2id = ctx.iss.set_index("Issue_Key").issue_id.to_dict()
        lk = ctx.byf.get("Link", ctx.empty)
        dep = dependency_events(lk)
        dep["other"] = dep.other_key.map(self.key2id)
        dep = dep.dropna(subset=["other"])
        dep["pred"] = np.where(dep.direction == "in", dep.other, dep.issue_id).astype(int)
        dep["succ"] = np.where(dep.direction == "in", dep.issue_id, dep.other).astype(int)
        self.deps = dep[dep.added][["pred", "succ", "ts"]]
        spr = ctx.byf.get("Sprint", ctx.empty)
        self.sprint_ch = spr[["Issue_ID", "Creation_Date", "To_Value"]]
        self.sp_ch = ctx.byf.get("Story Points", ctx.empty)[["Issue_ID", "Creation_Date"]]
        jira2series = sp.dropna(subset=["JiraID"]).set_index(["project_id", "JiraID"]).series
        jira2series = jira2series[~jira2series.index.duplicated()]
        proj = ctx.iss.set_index("issue_id").project_id
        mem = spr[spr.To_Value.notna()][["Issue_ID", "To_Value"]].copy()
        mem["jid"] = mem.To_Value.astype(str).str.split(",")
        mem = mem.explode("jid")
        mem["jid"] = pd.to_numeric(mem.jid.str.strip(), errors="coerce")
        mem["p"] = mem.Issue_ID.map(proj)
        mem["series"] = [jira2series.get((p_, j)) for p_, j in zip(mem.p, mem.jid)]
        self.series_issues = mem.dropna(subset=["series"]).groupby(["p", "series"]).Issue_ID.apply(set).to_dict()

    # ------------------------------------------------------------------ calendar
    def sprint_length(self, p):
        s = self.sprints[self.sprints.project_id == p].dropna(subset=["Start_Date", "End_Date"])
        d = (s.End_Date - s.Start_Date).dt.days
        d = d[(d > 2) & (d < 60)]
        return int(d.median()) if len(d) else 14

    def episode_starts(self, p, t_from, t_to):
        """(start, series) of real sprints starting inside the window, one team board per episode."""
        s = self.sprints[(self.sprints.project_id == p) & self.sprints.Start_Date.between(t_from, t_to)]
        s = s.dropna(subset=["Start_Date"]).sort_values("Start_Date")
        s = s[np.array([(p, x) in self.series_issues for x in s.series], dtype=bool)]
        return list(dict.fromkeys(zip(s.Start_Date.dt.normalize(), s.series)))

    # ------------------------------------------------------------------ helpers
    def _remaining_hours(self, ids, T0):
        iv = self.iv[self.iv.issue_id.isin(ids) & (self.iv.end > T0)]
        s = np.maximum(iv.start.values.astype("datetime64[s]"), np.datetime64(T0, "s"))
        e = iv.end.values.astype("datetime64[s]")
        full = weekday_hours(iv.start.values.astype("datetime64[s]"), e)
        part = weekday_hours(s, e)
        frac = np.where(full > 0, part / np.maximum(full, 1e-9), 1.0)
        return pd.Series(iv.hours.values * frac, index=iv.issue_id.values).groupby(level=0).sum()

    def _dev_hours(self, devs, bounds):
        """Real active hours per developer per period (WIP-adjusted, all their issues in scope)."""
        iv = self.iv[self.iv.assignee.isin(devs)]
        out = np.zeros((len(devs), len(bounds) - 1))
        di = {d: i for i, d in enumerate(devs)}
        s_all = iv.start.values.astype("datetime64[s]")
        e_all = iv.end.values.astype("datetime64[s]")
        full = weekday_hours(s_all, e_all)
        for k in range(len(bounds) - 1):
            a, b = np.datetime64(bounds[k], "s"), np.datetime64(bounds[k + 1], "s")
            s, e = np.maximum(s_all, a), np.minimum(e_all, b)
            ok = e > s
            part = np.where(ok, weekday_hours(np.where(ok, s, a), np.where(ok, e, a)), 0.0)
            h = iv.hours.values * np.where(full > 0, part / np.maximum(full, 1e-9), 0.0)
            np.add.at(out[:, k], iv.assignee.map(di).values.astype(int), h)
        return out

    def _wip_limits(self, devs, t_from, t_to):
        """Learnt concurrency: per developer, mean number of simultaneously active issues while active
        (sum of interval business hours / business hours covered by their union), before t_to."""
        iv = self.iv[self.iv.assignee.isin(devs) & (self.iv.start < t_to) & (self.iv.end > t_from)]
        out = {}
        for a, g in iv.groupby("assignee"):
            s = np.maximum(g.start.values.astype("datetime64[s]"), np.datetime64(t_from, "s"))
            e = np.minimum(g.end.values.astype("datetime64[s]"), np.datetime64(t_to, "s"))
            tot = weekday_hours(s, e).sum()
            o = np.argsort(s)
            s, e = s[o], e[o]
            union, cs, ce = 0.0, s[0], e[0]
            for a_, b_ in zip(s[1:], e[1:]):
                if a_ <= ce:
                    ce = max(ce, b_)
                else:
                    union += weekday_hours(np.array([cs]), np.array([ce]))[0]
                    cs, ce = a_, b_
            union += weekday_hours(np.array([cs]), np.array([ce]))[0]
            out[a] = tot / union if union > 0 else 1.0
        med = np.median(list(out.values())) if out else 1.0
        return np.array([max(int(np.ceil(out.get(d, med))), 1) for d in devs])

    def _due(self, ids, T):
        ch = self.sprint_ch[self.sprint_ch.Issue_ID.isin(ids)].copy()
        ch["tref"] = ch.Issue_ID.map(T)
        ch = ch[(ch.Creation_Date <= ch.tref) & ch.To_Value.notna()].sort_values("Creation_Date")
        last = ch.groupby("Issue_ID").To_Value.last()
        jira = pd.to_numeric(last.astype(str).str.split(",").str[-1], errors="coerce")
        ends = self.sprints.dropna(subset=["JiraID"]).set_index("JiraID").End_Date
        ends = ends[~ends.index.duplicated()]
        return jira.map(ends)

    # ------------------------------------------------------------------ build
    def build(self, p, T0, n_sprints, model, eff_plan, eff_true, cap_hist_sprints=8, series=None, max_tasks=150):
        if isinstance(T0, tuple):
            T0, series = T0
        L_cal = self.sprint_length(p)
        L = max(int(round(L_cal * 5 / 7)), 3)
        T0 = pd.Timestamp(T0)
        T_end = T0 + pd.Timedelta(days=L_cal * n_sprints)
        bounds = [T0 + pd.Timedelta(days=L_cal * k) for k in range(n_sprints + 1)]
        F = self.frame
        fr = F[(F.project_id == p) & F.first_sprint_at.notna() & (F.first_sprint_at <= T_end)
               & F.resolved_at.notna() & (F.resolved_at > T0)]
        if series is not None:
            fr = fr[fr.index.isin(self.series_issues.get((p, series), set()))]
        if len(fr) < 4 or len(fr) > max_tasks:
            return None
        rem = self._remaining_hours(set(fr.index), T0)
        entry = fr.first_sprint_at.clip(lower=T0)
        # estimates as of entry (planner view)
        feats = self.ctx.features_as_of(entry)
        sp_fill = float(F[(F.project_id == p) & (F.resolved_at < T0) & F.sp.notna()].sp.median() or 3.0)
        feats["sp_missing"] = feats.sp.isna() | (feats.sp <= 0)
        feats.loc[feats.sp_missing, "sp"] = sp_fill
        feats["sp"] = feats.sp.clip(0.5, 100)
        pred = model.predict(feats, intervals=False).set_index("issue_id")
        prof = [c for c in pred.columns if c.startswith("profile")]
        cl = pred[prof].values.argmax(1) if prof else np.zeros(len(pred), int)
        cluster = pd.Series(cl, index=pred.index)
        due_at = self._due(set(fr.index), entry)
        devs = sorted(set(fr.Assignee_ID.dropna().astype(int)))
        if not devs:
            return None
        K = max(len(prof), 1)
        eta_p, eta_t = eff_plan.eta(devs), eff_true.eta(devs)
        if eta_p.shape[1] < K:
            eta_p = np.pad(eta_p, ((0, 0), (0, K - eta_p.shape[1])), constant_values=1.0)
            eta_t = np.pad(eta_t, ((0, 0), (0, K - eta_t.shape[1])), constant_values=1.0)
        di = {d: i for i, d in enumerate(devs)}

        def task(i):
            r = fr.loc[i]
            k = int(cluster.get(i, 0))
            ass = r.Assignee_ID
            eta_real = eta_t[di[int(ass)], k] if pd.notna(ass) and int(ass) in di else 1.0
            true_h = max(float(rem.get(i, 0.0)), 0.25)
            d = due_at.get(i)
            due_s = np.inf if d is None or pd.isna(d) else max(float((d - T0).days) / L_cal - 1, 0.0)
            preds = self.deps[(self.deps.succ == i) & (self.deps.ts <= entry[i])].pred.tolist()
            return Task(id=str(i), req=str(i), cluster=k, weight=float(_prio(feats.set_index("issue_id").priority_t.get(i))),
                        due_sprint=due_s, preds=[str(x) for x in preds if x in fr.index],
                        est_hours=float(pred.hours[i]), true_ref_hours=true_h * eta_real)

        initial = [task(i) for i in fr.index if fr.first_sprint_at[i] <= T0]
        events = []
        day_of = lambda ts: int(np.busday_count(np.datetime64(T0.date()), np.datetime64(pd.Timestamp(ts).date())))
        for i in fr.index:
            if fr.first_sprint_at[i] > T0:
                events.append((day_of(fr.first_sprint_at[i]), "ADD", task(i)))
            if fr.Resolution[i] not in DONE_RESOLUTIONS and fr.resolved_at[i] <= T_end:
                events.append((day_of(fr.resolved_at[i]), "DROP", str(i)))
        sp_ch = self.sp_ch[self.sp_ch.Issue_ID.isin(fr.index) & self.sp_ch.Creation_Date.between(T0, T_end)]
        if len(sp_ch):
            sp_ch = sp_ch[sp_ch.Creation_Date > sp_ch.Issue_ID.map(entry)]
            if len(sp_ch):
                Tm = pd.Series(sp_ch.Creation_Date.values, index=sp_ch.Issue_ID.values)
                Tm = Tm[~Tm.index.duplicated(keep="last")]
                fm = self.ctx.features_as_of(Tm)
                fm["sp"] = fm.sp.fillna(sp_fill).clip(0.5, 100)
                pm = model.predict(fm, intervals=False)
                for i, h, ts in zip(pm.issue_id, pm.hours, Tm.loc[pm.issue_id].values):
                    events.append((day_of(ts), "MODIFY", (str(i), float(h))))
        events.sort(key=lambda e: (e[0], {"ADD": 0, "MODIFY": 1, "DROP": 2}[e[1]]))
        hist_bounds = [T0 - pd.Timedelta(days=L_cal * k) for k in range(cap_hist_sprints, -1, -1)]
        hist = pd.DataFrame(self._dev_hours(devs, hist_bounds), index=devs)
        cap_plan = CapacityForecaster().fit(hist).forecast(devs, n_sprints)
        cap_plan = np.where(cap_plan <= 0, max(np.median(cap_plan[cap_plan > 0]) if (cap_plan > 0).any() else 8.0, 1.0), cap_plan)
        cap_true = self._dev_hours(devs, bounds)
        wip = self._wip_limits(devs, hist_bounds[0], T0)
        return EpisodeData(tasks=initial, events=events, sprint_days=L, n_sprints=n_sprints,
                           dev_names=[str(d) for d in devs], eta_plan=eta_p, eta_true=eta_t,
                           cap_true=cap_true, cap_plan=cap_plan, kappa=eff_plan.kappa, handover=eff_plan.handover,
                           wip_limit=wip), \
            dict(project=p, T0=T0, series=series, L_cal=L_cal, devs=devs, n_tasks=len(fr), entry=entry, fr=fr, rem=rem)


def _prio(v):
    from .features import priority_ord
    return priority_ord(v)


def fit_efficiency(model, rows, frame_full=None):
    """Execution-model training rows: actual effort vs the estimator's reference prediction."""
    pred = model.predict(rows, intervals=False)
    prof = [c for c in pred.columns if c.startswith("profile")]
    k = pred[prof].values.argmax(1) if prof else np.zeros(len(pred), int)
    r = rows.reset_index(drop=True)
    wip = r.groupby("Assignee_ID").issue_id.transform("size").fillna(1)    # placeholder replaced below
    # concurrent WIP: number of the assignee's issues whose active span overlaps this issue's span
    span = r[["Assignee_ID", "work_start", "work_end"]]
    wip = np.zeros(len(r))
    for a, g in span.groupby("Assignee_ID").groups.items():
        s = r.loc[g, "work_start"].values
        e = r.loc[g, "work_end"].values
        wip[g] = ((s[:, None] < e[None]) & (e[:, None] > s[None])).sum(1) - 1
    df = pd.DataFrame(dict(dev=r.Assignee_ID, cluster=k, actual=r.effort_h, ref=pred.hours.values, wip=wip,
                           reassigned=(r.n_assignees > 1).astype(float), familiarity=r.familiarity,
                           tenure=r.tenure_days))
    return EfficiencyModel().fit(df, max(len(prof), 1))


def backtest(ep_data, info):
    """E4: execute the team's real assignment with real effort and capacity; compare with reality."""
    fr, T0, L_cal = info["fr"], info["T0"], info["L_cal"]
    tasks = ep_data.tasks + [e[2] for e in ep_data.events if e[1] == "ADD"]
    ids = [t.id for t in tasks]
    idx = {i: k for k, i in enumerate(ids)}
    devs = info["devs"]
    di = {d: i for i, d in enumerate(devs)}
    dev = np.array([di.get(int(fr.Assignee_ID[int(i)]), 0) if pd.notna(fr.Assignee_ID[int(i)]) else 0 for i in ids])
    ws = fr.work_start.reindex([int(i) for i in ids])
    sprint = np.array([max(int((w - T0).days // L_cal), 0) if pd.notna(w) else 0 for w in ws])
    inst = ScheduleInstance(ids=ids, req=ids, remaining=np.array([t.true_ref_hours for t in tasks]),
                            cluster=np.array([t.cluster for t in tasks]), weight=np.array([t.weight for t in tasks]),
                            due=np.array([t.due_sprint if np.isfinite(t.due_sprint) else 1e9 for t in tasks]),
                            preds=[[idx[p] for p in t.preds if p in idx] for t in tasks], eta=ep_data.eta_true,
                            capacity=np.maximum(ep_data.cap_true, 1e-3), dev_names=ep_data.dev_names,
                            kappa=ep_data.kappa)
    true_hours = inst.cost[np.arange(inst.n), dev]
    out = execute(inst, sprint, dev, true_hours, inst.capacity, wip_limit=ep_data.wip_limit)
    actual = np.array([(fr.resolved_at[int(i)] - T0).days / L_cal for i in ids])
    sim = out["completion"]
    within = actual <= ep_data.n_sprints
    return dict(n=int(within.sum()), mae_sprints=float(np.abs(sim - actual)[within].mean()) if within.any() else np.nan,
                same_sprint=float((np.floor(sim) == np.floor(actual))[within].mean()) if within.any() else np.nan,
                rank_corr=float(pd.Series(sim[within]).corr(pd.Series(actual[within]), method="spearman"))
                if within.sum() > 3 else np.nan)
