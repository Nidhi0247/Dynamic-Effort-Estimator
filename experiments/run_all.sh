#!/usr/bin/env bash
# Full study: E1/E2 accuracy + ablations, E3-E5 dynamic loop, E6/E7 scheduling.  Logs in experiments/results/logs.
set -e
cd "$(dirname "$0")"
PY=../venv/Scripts/python.exe
[ -x "$PY" ] || PY=python
mkdir -p results/logs
$PY e1_accuracy.py --backend "${BACKEND:-minilm}" 2>&1 | grep -v -i warning | tee results/logs/e1.log
$PY e3_dynamic.py  --backend "${BACKEND:-minilm}" --jobs "${JOBS:-16}" 2>&1 | grep -v -i warning | tee results/logs/e3.log
$PY e6_scheduling.py --backend "${BACKEND:-minilm}" --jobs "${JOBS:-16}" 2>&1 | grep -v -i warning | tee results/logs/e6.log
