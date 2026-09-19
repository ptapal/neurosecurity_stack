# analyses

Checks behind specific paper claims. Run from the repo root as `python -m analyses.<script>`.

| Script | What | Why |
|---|---|---|
| `thm61_check.py` | DP-FedProx-EEG training against the Theorem 6.1 bound, writes `results/<cohort>_pillar1_v2.json` | test whether the convergence bound holds and how loose it is |
| `ablation_fk_sweep.py` | sweep of compromised clients (1, 4, 7, 10 of 15) with and without the Pillar 3 gate, writes `results/pillar1_fk_sweep_v2.json` | show poisoning damage and how much the gate removes |
| `bound_check.py <cohort>` | Theorem 8.1 power, Neyman-Pearson bound and unwhitened variant against measured detection power | test whether the CLT power formula predicts the measurements |
| `hw_check.py <cohort> <sigma>` | Hanson-Wright lower bound on detection power | non-asymptotic alternative to Theorem 8.1 |
| `qq_gaussian_check.py` | chi-square QQ-plot and Shapiro-Wilk rates of squared Mahalanobis distances | check the Gaussian assumption behind Theorem 8.1 |
| `autocorr_check.py` | lag-1 autocorrelation of the alpha-power embedding | check the independence assumption in the group-privacy accounting |
| `subsample_sensitivity.py` | Pillar 2 and 3 sweep redrawn on 10 alternative 15-subject samples per cohort, writes `results/subsample_sensitivity.json` | show the results do not depend on which 15 subjects were drawn |
| `joint_optimizer.py` | Nelder-Mead search over injection strength, frequency and channel fraction against Pillars 2 and 3 | probe a joint adaptive adversary |

Caveats:

- `autocorr_check.py` uses the conditions of Table 11 (first 200 or 150 trials per subject, mean over channels) on the chronological sequence: Zhang, Wang and COG-BCI are loaded with `ordered=True` (the default pipeline order is unchanged). Cho is released as one block per class with no index of the original (randomized) order, so its value is the lag-1 autocorrelation of each class block, averaged over the two blocks. Won and Zhang epochs (0.8 s) are cut around stimuli about 0.1 s apart, so consecutive epochs overlap and correlate by construction; the `non_overlap` column restricts the statistic to pairs of epochs that do not overlap, and is the one to report for those cohorts.
- `ablation_fk_sweep.py` seeds its draws from Python's salted `hash()`. Set `PYTHONHASHSEED` to repeat the same draws.
- `bound_check.py` prints a `paper_theory` column that is the earlier closed form from `detect.theoretical_power_bound`, not Theorem 8.1.
- `joint_optimizer.py` runs one cohort (`zhang`) with 4 starts; the repeated draws in Appendix E are not in this script.
