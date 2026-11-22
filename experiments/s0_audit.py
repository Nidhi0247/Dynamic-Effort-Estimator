"""Stage 0 data audit: what each TAWOS project can support, and which projects pass the gate.

Writes results/s0_audit.csv and prints the inclusion funnel.
Gate (effort-labelled core): >= 150 resolved issues with SP > 0 and logged effort, >= 10 sprints.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynest.realdata.store import Store  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
MIN_ISSUES, MIN_SPRINTS = 150, 10


def main():
    st = Store()
    iss = st.issues
    proj = st.projects.set_index("ID")
    spr = st.sprints
    cl = st.changelog(fields=["Sprint", "timespent", "Story Points", "Link", "description", "assignee"])
    links = st.links
    resolved = iss["Resolution_Date"].notna()
    has_sp = iss["Story_Point"].fillna(0) > 0
    has_ts = iss["Timespent"].fillna(0) > 0
    has_ip = iss["In_Progress_Minutes"].fillna(0) > 0
    rows = []
    for pid, g in iss.groupby("project_id"):
        ids = set(g.issue_id)
        r, sp, ts, ip = resolved[g.index], has_sp[g.index], has_ts[g.index], has_ip[g.index]
        c = cl[cl.Issue_ID.isin(ids)]
        rows.append(dict(
            project_id=pid, key=proj.loc[pid, "Project_Key"], name=proj.loc[pid, "Name"],
            issues=len(g), resolved=int(r.sum()), sp_share=float(sp.mean()),
            timespent_share=float(ts.mean()), inprogress_share=float(ip.mean()),
            core_timespent=int((r & sp & ts).sum()), core_inprogress=int((r & sp & ip).sum()),
            sprints=int((spr.project_id == pid).sum()),
            sprint_changes=int((c.Field == "Sprint").sum()),
            timespent_changes=int((c.Field == "timespent").sum()),
            sp_changes=int((c.Field == "Story Points").sum()),
            link_rows=int(links.src.isin(ids).sum()),
            assignees=int(g.Assignee_ID.nunique()),
            first=g.Creation_Date.min(), last=g.Creation_Date.max(),
            median_timespent_h=float((g.Timespent[ts[ts].index] / 3600).median()) if ts.any() else np.nan,
        ))
    a = pd.DataFrame(rows).sort_values("core_timespent", ascending=False)
    a["gate_timespent"] = (a.core_timespent >= MIN_ISSUES) & (a.sprints >= MIN_SPRINTS)
    a["gate_inprogress"] = (a.core_inprogress >= MIN_ISSUES) & (a.sprints >= MIN_SPRINTS)
    OUT.mkdir(exist_ok=True)
    a.to_csv(OUT / "s0_audit.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(a[["project_id", "key", "issues", "resolved", "sp_share", "timespent_share", "core_timespent",
             "core_inprogress", "sprints", "timespent_changes", "link_rows", "assignees", "median_timespent_h",
             "gate_timespent", "gate_inprogress"]].to_string(index=False))
    print("\nFunnel:")
    print(f"  projects                         {len(a)}")
    print(f"  >= {MIN_SPRINTS} sprints                      {(a.sprints >= MIN_SPRINTS).sum()}")
    print(f"  gate (logged effort, Timespent)  {a.gate_timespent.sum()}")
    print(f"  gate (In_Progress proxy)         {a.gate_inprogress.sum()}")
    return a


if __name__ == "__main__":
    main()
