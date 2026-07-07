## PhySDV: Physics-Grounded Data Generation and Risk-Aware Learned Offloading for QoS-Constrained Perception in Software-Defined Vehicles

Features

1. **Risk-aware quantile QoS gating** — [src/predictor/quantile.py](vscode-webview://1vncj99q993r856inrl72drj524r55p2h51fll0o0upl0ev57opa/src/predictor/quantile.py): p90 pinball-loss RTT regressors; feasibility gated on the upper quantile instead of the mean. Measured: worst-scenario latency compliance  **94.0% → 96.6%** , dense-urban  **94.6% → 97.0%** , at *unchanged* accuracy (cost = +3.3pp WAN share). This is the headline mechanism.
2. **DwellTimeStabilizer with safety bypass** — [optimizer.py](vscode-webview://1vncj99q993r856inrl72drj524r55p2h51fll0o0upl0ev57opa/src/offloading/optimizer.py): hysteresis cuts switching **142 → 6 per 1k frames (−96%)** while outage fallback stays zero-delay (infeasibility bypasses the filter).
3. **Three-level validation protocol** (structural / predictive / task-level) — the methodological contribution: path-loss exponents recovered from the data to ≤1.6% error; closed-loop  **oracle agreement up to 98.8%** .
4. **Predictor deployment study** (your "possible predictor") — HistGB beats the repo's RandomForest: **6.1 ms MAE, 1.3 MB, 3.8 ms** vs 10.5 ms / 926 MB / 14.3 ms.
5. **Emergent crossover finding** — in dense urban, MEC beats cloud on only 33% of frames (Always-MEC collapses to 53.5% compliance), which *forces* per-frame learned prediction. Great reviewer defense for "why not a static policy."
