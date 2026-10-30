# Choices

Non-obvious choices in `laser_reservoir.ipynb`, and why they were made.

## Data (shared by ESN and laser reservoir)

| Choice | Value | Why |
|---|---|---|
| Random seed for NARMA10 | 42 | Both reservoirs get exactly the same input and target. |
| Target alignment | d(k) = y(k+1) | y(k+1) depends only on inputs up to u(k). So after seeing u(k), the model predicts y(k+1). |
| Washout | first 200 steps discarded | The reservoir starts at zero and y starts at 0. Those first steps do not show normal behaviour. |
| Split | 3000 train / 800 validation / 1000 test | Settings are chosen on validation. Test is used once, for the final score, so the score is not tuned to the test data. |
| Final training set | train + validation (3800 steps) | After settings are fixed, more training data gives a better readout. |
| Blow-up check | stop if y is not finite or y > 10 | NARMA10 can diverge for rare inputs. |
| Metric | NMSE (Appeltant reports NRMSE = √NMSE) | The assignment asks for NMSE. |

## ESN

| Choice | Value | Why |
|---|---|---|
| Number of nodes N | 100 | Same as Appeltant's "traditional RC" comparison. |
| Connection density | 10% of W nonzero | Standard for ESNs. Not tuned. |
| Spectral radius ρ, input scaling s, ridge α | chosen by grid search on validation (result: ρ = 0.9, s = 0.1, α = 1e-8) | These three most affect the score. Picking them on validation is a fair way to tune the baseline. |
| Bias b | uniform in [−0.1, 0.1] | Places nodes at different points on tanh. In a quick test it lowered NMSE (≈0.11 → ≈0.09). |
| Readout features | [x(k), u(k), 1] | Raw input and a constant are standard in ESN readouts. Use the same extra features for the laser reservoir so the comparison is fair. |
| Number of random reservoirs | 3 for tuning, 10 for final score | One random reservoir can be lucky or unlucky. Report mean ± std. |

## Open points

- Our ESN gets test NMSE ≈ 0.067 (NRMSE ≈ 0.26). Appeltant's traditional RC reports NMSE ≈ 0.032. Their reservoir design and training length are not fully given, so this gap is not resolved.
