# DynEst-R: trained, real-data dynamic effort estimation and sprint re-planning

DynEst-R re-estimates effort whenever a requirement changes and decides when to re-plan the sprint.
It does this with a pipeline in which **every component is trained on real project history**.
There is no synthetic data, and no hand-set rule base, weight, threshold or constant.

Data comes from the [TAWOS](https://github.com/SOLAR-group/TAWOS) Jira dataset: 39 projects, 458k issues and 9.25M changelog entries.

## Quick start
```bash
python -m venv venv && venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
# 1. download TAWOS (https://doi.org/10.5522/04/21308124) and unzip TAWOS.sql into data/raw/
python -m dynest.realdata.sqldump data/raw/TAWOS.sql data/tawos   # ~12 min; no MySQL needed
python experiments/s0_audit.py                                   # Stage-0 data audit / gate
python -m pytest -q tests                                        # 16 tests (method + real-data leakage)
python run_demo.py                                               # replay a held-out project window
streamlit run app.py                                             # dashboard
bash experiments/run_all.sh                                      # full study (E1-E7), results in experiments/results
```
Add `--smoke --backend tfidf` to any `experiments/e*.py` script for a quick end-to-end check.

## Pipeline (what is trained, on what)
| Stage | Component | Trained from | Code |
|---|---|---|---|
| Data | Streaming dump-to-parquet converter; as-of snapshots by replaying the changelog | — | `dynest/realdata/sqldump.py`, `store.py` |
| Effort label | Measured **active work hours**: business hours in work statuses, shared across the assignee's concurrent WIP | status history | `dynest/realdata/effort.py` |
| S1 | As-of features (counts, story points, familiarity, …) plus MiniLM text embedding with PCA | training window | `dynest/features.py`, `text.py` |
| S2 | Learned attributes: `clarity_hat`, `integration_hat`, and component-topic profiles (NMF) | later rewrites, questions, links, components | `dynest/aux_models.py` |
| S3 | SOM clusters with a shrunk cluster prior | log hours per SP | `dynest/som.py` |
| S4 | ANFIS complexity score (out-of-fold stacked; online RLS updates) | log hours per SP | `dynest/anfis.py` |
| S5 | **Learned Mamdani FIS**: Wang–Mendel rules plus GA-tuned strong fuzzy partitions | log hours per SP | `dynest/learned_mamdani.py` |
| — | Back-transform shift that minimizes validation MAE | validation window | `dynest/calibration.py` |
| S6 | Normalized, Mondrian, rolling **conformal intervals** (requirement and project level) | validation residuals | `dynest/conformal.py` |
| S7 | Execution model: developer × cluster efficiency, context switching κ, handover, capacity forecasts, WIP limits | resolved issues and activity | `dynest/execution.py`, `replay.py` |
| S8 | Replay of real change events. Repair below the trigger. Trigger is either a **learned** (logistic, branch-simulation labels) or a tuned τ | validation-window episodes | `dynest/sim/`, `dynest/tracker.py` |
| S9 | Random-key scheduler (dependencies, locks, partial progress, capacity) with four objectives. GA, PSO, NSGA-II, MOPSO; compromise plan chosen by **preference weights learned** from the team's real plans | real plans | `dynest/scheduling/` |
| S10 | Explanations: fired learned rules, aggregated output, cluster, intervals | — | `app.py` |

## Experiments (`experiments/`)
| Script | Question |
|---|---|
| `s0_audit.py` | Which projects have usable effort data (gate: ≥150 rows, ≥10 sprints) |
| `e1_accuracy.py` | E1 accuracy against baselines B0–B7, and E2 ablations A1–A5. Uses 3 rolling origins with Friedman, Wilcoxon/Holm and A12 tests, with projects as the unit of analysis |
| `e3_dynamic.py` | E3 tracking error, E4 execution-model back-test, and E5 policies (static, sprint-boundary, τ grid, tuned τ, learned trigger, greedy) |
| `e6_scheduling.py` | E7 preference learning, then E6 optimizers (score, hypervolume, IGD+, Dirichlet robustness) |

## Honest data notes
- **Effort is not logged in TAWOS.** `Timespent` is usable in only 2 of 39 projects, with a median of about 2 minutes per issue. Effort is therefore a *measured proxy*: active work hours, adjusted for work in progress. Report it as such.
- **Decision time.** The estimate is taken one second before work starts. TAWOS's `Estimation_Date` is a creation-time artefact and is not used.
- **Counterfactual schedules are model-based.** Policy outcomes come from replaying real events and real capacities with the trained execution model. That model is validated by the E4 back-test, which replays the team's real assignment.
- **Leakage controls.** All splits are per project and in time order, with a one-sprint gap. Every feature is computed as of the decision time, and tests in `tests/test_realdata.py` check this.
