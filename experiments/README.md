# Paper experiments

Seeded, single-command scripts that regenerate every number and figure in
`new_paper/main.tex`. Run from the repository root:

```bash
python experiments/exp1_physics_validation.py   # V1: path-loss recovery, per-scenario stats
python experiments/exp2_model_comparison.py     # V2: model-family study (also trains models/predictor.pkl)
python experiments/exp2b_refined_metrics.py     # V2: reachable-row metrics + HistGB bundle
python -m src.predictor.quantile                # q90 quantile models + coverage
python experiments/exp3_closed_loop.py          # V3: closed loop, oracle, baselines (RF default)
python experiments/exp3_closed_loop.py models/predictor_histgb.pkl _histgb
python experiments/exp4_quantile_gating.py      # V3: risk-aware q90 gating + outage traces
python experiments/gen_figures.py               # PNG figures for the paper
```

Results are written as `results_exp*.json` next to the scripts.
