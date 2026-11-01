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

## Laser on its own (Step 2)

| Choice | Value | Why |
|---|---|---|
| Equations | Cho et al. PRL 2016, Eqs. 1–2, unchanged | These are the lab's own model of its laser particles. |
| Time unit | τ = 1 (carrier lifetime) | All times are then multiples of τ. With τ ≈ 2 ns for perovskite, 1 time unit = 2 ns. |
| Photon lifetime τ_c | 0.01 τ | Photons must leave much faster than carriers decay. Real values may be shorter still. 0.01 keeps the simulation fast, and it is a parameter we can vary later. |
| β | 1e-3 | Close to the paper's fitted 1.3e-3. Gives a clear threshold. |
| Normalized outputs | p = P/P_th, n = N₁/N_th, L = β·q | Threshold is at p = 1 and n = 1 for any τ_c and β. L is the paper's Eq. 3 quantity. |
| Solver | 4th-order Runge–Kutta, fixed step dt = 1e-3 τ (τ_c/10) | A fixed step makes it easy to hold the pump constant within a step, which Step 3 needs. Halving dt changes the output by less than 1e-8 (relative). |
| Pump within a step | constant | The pump in Step 3 is piecewise constant, so this is exact there. |
| Noise | none | Keeps the first results easy to interpret. Spontaneous emission noise can be added later. |
| Starting state | N₁ = 0, q = 0, or the steady state for the small-step test | The β·N₁/τ term creates the first photons, so q = 0 is a valid start. |

## Laser as a reservoir (Step 3)

| Choice | Value | Why |
|---|---|---|
| Two versions | A: no feedback. B: delay loop | A tests the laser particle alone, which is closest to Fred's question. B tests the particle inside Appeltant's design. A real particle has no loop; adding one needs extra hardware (detector, delay, pump modulator). |
| Where the input goes | the pump: p = p_bias + γ·mask_i·u(k) (+ η·L(t − delay)) | The pump is the only input the paper's model has. |
| Number of virtual nodes N | 100 | Same as the ESN, so both readouts get the same number of features. |
| Mask | uniform random in [−1, 1], fixed | In an early search, a ±1 mask (Appeltant's choice) was a little worse. That search only tried γ ≥ 0.5. |
| Feedback | η·L, added to the pump, delay = N·θ | Same structure as Appeltant. A delay of N+1 slots was also tried in the early search (γ ≥ 0.5 only); it was not better. |
| Delayed value within a solver step | held constant | The step (0.005 τ) is far shorter than the delay. |
| Negative pump | clipped to 0 | A pump cannot be negative. Only happens at large γ. |
| Recording | output L at the end of each slot | One number per slot, like sampling a photodetector. |
| Readout | rescale recordings to mean 0 and std 1 (training statistics), then [recordings, u(k), 1] and ridge | Same features as the ESN. Rescaling makes the ridge penalty act the same way as for the ESN. |
| Solver step | 0.005 τ (Step 2 used 0.001 τ) | 5× faster. Checked by halving and quartering the step: the reservoir output changes by up to 0.06%, and the test NMSE by less than 0.001. Same check for τ_c = 0.003 with step 0.001. |
| Solver speed | compiled with `numba` | One run is millions of steps. numba is installed in Colab by default. |
| Settings search | A: p_bias 0.7–1.3, γ 0.5–8, θ 0.005–0.02 (60 settings). B: p_bias 1.1–1.6, γ 0.05–0.5, θ 0.2–0.5, η 0.3–0.9 (72 settings). Scored on validation. | Same method as the ESN. The grid was widened twice because the best settings kept landing on its edge. Lowering γ from ≥ 0.5 to 0.05 changed Laser B from 0.46 to 0.115. |
| Smallest γ | 0.05, not lower, even though it is the grid edge | The model has no noise, so smaller γ would only exploit perfect readings. Noise in a real laser would hide pump changes this small. |
| Smallest θ for A | 0.005 τ, the grid edge | It equals one solver step, so θ cannot go lower without a smaller step. A stays far worse than the linear memory reference anyway. |
| Test repeats | 5 random masks | Fewer than the ESN's 10, to save time. The spread is reported. |
| Linear memory reference | ridge on the last 10 raw inputs | Our own version of Appeltant's shift-register baseline, on the same data. |
| Nonlinearity test | recover ũ(k)·ũ(k−j), with ũ = u − mean(u) | Without removing the mean, memory alone scores well on this test. |
| Threads | one thread per numpy call | Many threads on tiny matrices made the notebook about 20× slower. |

## Exploration: photon lifetime τ_c (Section 4)

| Choice | Value | Why |
|---|---|---|
| Parameter | τ_c | Memory limited the laser in Section 3, and τ_c sets the laser's response speed. It is a particle-design property, so the result is useful to the lab. Not chosen: β (tests nonlinearity), p_bias (already in the search), N (does not directly test memory). |
| Values | 0.003, 0.01, 0.03, 0.1, 0.3 τ | A 100-fold range around the main value 0.01. Real perovskite particles are probably below 0.01. |
| Fixed | β = 1e-3, N = 100, data, split, masks | Change one thing at a time. |
| Threshold | pumps given as p = P/P_th | P_th = 1/(β·τ_c) changes with τ_c. Using p keeps the operating point the same. |
| Drive settings | re-tuned for each τ_c with the Section 3 grid | Each particle gets its best drive settings, so we compare particles, not one fixed drive. |
| Solver step | 0.001 τ for τ_c = 0.003, else 0.005 τ | The step must stay well below τ_c. |
| Step test | pump 1.05 → 1.15, settled = within 1% | Near the operating point of Laser B. |
| Saved results | `results/tau_c_sweep.json` | The search takes about 4 minutes. The notebook loads the file; delete it to recompute. |

## Noise (Section 5)

| Choice | Value | Why |
|---|---|---|
| Noise source | the laser's own quantum noise: every process (pump, stimulated emission, spontaneous emission into the mode, other spontaneous emission, photon loss) has Poisson statistics, variance = rate × dt | It follows from the physics, with no new free parameter. Its size is set by the photon number. |
| Pump noise | included (Poisson) | A pump laser's photons also arrive at random. A quieter pump would help a little. |
| Detector noise | not included | Quantum noise alone already dominates. |
| Solver | Euler–Maruyama, step 0.0025 τ (τ_c / 4), raw numbers N₁ and q | Standard method for equations with random terms. Without noise it gives the same score as the RK4 solver (0.106 vs 0.107). |
| Negative numbers | N₁ and q clipped at 0 | Random steps can push small numbers below 0. |
| Re-tuning | new grid searched with noise on: A 48 settings, B 150 settings (γ up to 1.0, p_bias up to 3.0), for β = 1e-2 to 1e-5 | With noise, a larger input may win. Same validation method as Section 3. |
| Noise seed | the same as the mask seed | Each test repeat gets its own mask and its own noise. |
| Search location | `scripts/noise_search.py`, results in `results/noise_search.json` | It takes about 5 minutes with 8 processes, too slow to rerun inside the notebook. Same data, split, masks, solver and readout as the notebook. |
| Recording | value at the end of the slot (as before); slot average checked as an alternative | A real detector integrates light over the slot. Averaging helped a little (0.649 → 0.544) without changing the conclusion. |

## Open points

- Our ESN gets test NMSE ≈ 0.067 (NRMSE ≈ 0.26). Appeltant's traditional RC reports NMSE ≈ 0.032. Their reservoir design and training length are not fully given, so this gap is not resolved.
- With γ ≥ 0.5, stronger feedback (η = 0.6–0.9) gave worse scores. With γ = 0.05, η = 0.6–0.9 is best at large τ_c. Not fully explained.
- With the laser's own noise, Laser B scores 0.61–0.71 (Section 5), worse than the linear memory reference. Untested ways round this: many more photons, a pump far above threshold, averaging repeated runs or many particles, or a task needing less memory.
- In the noise search, some best settings sit at the grid edge (γ = 1.0, p_bias = 1.1).
- At τ_c = 0.3, several best settings sit at the grid edge.
