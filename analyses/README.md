# analyses

Checks behind specific paper claims. Run from the repo root as `python -m analyses.<script>`.

| Script | What | Why |
|---|---|---|
| `thm61_check.py <cohort>` | DP-FedProx-EEG training against the convergence theorem bound, writes `results/<cohort>_convergence.json` | test whether the convergence bound holds and how loose it is |
| `ablation_fk_sweep.py` | sweep of compromised clients (1, 4, 7, 10 of 15) with and without the update gate, writes `results/fk_sweep.json` | show poisoning damage and how much the gate removes |
| `adaptive_poison.py [--cohorts ...]` | sign-flip vs norm-matched sign-flip, gate on/off, writes `results/adaptive_poison.json` | gate performance against a non-adaptive attacker |
| `adaptive_gate_attack.py [--cohorts ...]` | gate-aware attacker that scales its update to the largest value the gate accepts, writes `results/adaptive_gate.json` | gate performance against a worst-case attacker |
| `bound_check.py <cohort>` | detection theorem power, Neyman-Pearson bound and unwhitened variant against measured detection power | test whether the CLT power formula predicts the measurements |
| `hw_check.py <cohort> <sigma>` | Hanson-Wright lower bound on detection power | non-asymptotic alternative to detection theorem |
| `qq_gaussian_check.py` | chi-square QQ-plot and Shapiro-Wilk rates of squared Mahalanobis distances | check the Gaussian assumption behind detection theorem |
| `autocorr_check.py` | lag-1 autocorrelation of the alpha-power embedding | check the independence assumption in the group-privacy accounting |
| `subsample_sensitivity.py [--draws 10] [--cohorts ...]` | detection sweep redrawn on 10 alternative 15-subject samples per cohort, writes `results/subsample_sensitivity.json` | show the results do not depend on which 15 subjects were drawn |
| `realisations.py [--cohorts ...]` | 3 realisations per cohort at native and 0.8s window length, writes `results/realisations.json` | mean/std of the detection numbers over subject-sample and pipeline randomness |
| `strength_sweep.py [--cohorts ...]` | detection power at 3 injection strengths, 3 realisations per cohort, writes `results/strength_sweep.json` | check the inverted-U in injection strength |
| `joint_optimizer.py` | Nelder-Mead search over injection strength, frequency and channel fraction against both checks | probe a joint adaptive adversary |
| `joint_frontier.py [--cohorts ...]` | joint_optimizer repeated over 3 draws, minimum combined detection, writes `results/joint_frontier.json` | the joint-adversary numbers reported in the appendix |
| `epsilon_accounting.py` | formal RDP accounting for both released quantities, writes `results/epsilon_accounting.json` | the classical epsilon labels in the main text are outside their valid domain; this is the real accountant |
| `lee_label_order.py` | label vs trial order and block balance for Lee, writes `results/lee_label_order.json` | check Lee's labels are not block-confounded the way Cho's are |
| `summarise_clean.py` | collects every result above into `results/paper_numbers.json` | one file to check |

Caveats:

- `autocorr_check.py`: Cho is released as one block per class with no index of the original (randomized) order, so its value is the lag-1 autocorrelation of each class block, averaged over the two blocks. Won and Zhang epochs (0.8 s) are cut around stimuli about 0.1 s apart, so consecutive epochs overlap and correlate by construction; the `non_overlap` column restricts the statistic to pairs of epochs that do not overlap, and is the one to report for those cohorts.
- `bound_check.py` prints a `paper_theory` column that is the earlier closed form from `detect.theoretical_power_bound`, not detection theorem.
- `joint_optimizer.py` runs one cohort (`zhang`) with 4 starts; the repeated draws reported in the appendix are `joint_frontier.py`, not this script.
