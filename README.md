# Laser Particles as a Physical Reservoir

A simulation study. It tests whether one laser particle can work as a reservoir in reservoir computing. The laser is compared with a standard software reservoir, the Echo State Network (ESN).

## Background

- **Reservoir computing.** A fixed system turns each input into many numbers. Only a final weighted sum of these numbers is trained. The fixed system is the **reservoir**.
- **ESN.** A software reservoir: a recurrent neural network with fixed random connections.
- **Laser particle.** A small laser. Its light output responds to its pump (the light that powers it). The model is the rate equations of Cho et al. (2016).
- **NARMA10.** A standard test task. Each target depends on the last 10 inputs, so the reservoir needs memory and nonlinearity.
- **NMSE.** The test error: mean squared error divided by the variance of the target. 0 is perfect. 1 is no better than always predicting the mean.

The input enters the laser through its pump. Each input step is cut into 100 short time slots, and the laser output is recorded once per slot. Two versions are tested:

- **Laser A:** the particle alone.
- **Laser B:** the particle with a delay loop. Part of its earlier light is added back to the pump, as in Appeltant et al. (2011).

## Main results

Test NMSE on NARMA10 (lower is better):

| Model | NMSE |
|---|---|
| ESN, 100 nodes | 0.067 |
| Laser B, no noise | 0.115 |
| Linear memory reference (last 10 inputs, no reservoir) | 0.326 |
| Laser A, no noise | 0.648 |
| Laser B, with the laser's own quantum noise | 0.67 |

- Laser A has too little memory. Its memory lasts about 2 input steps. NARMA10 needs 9.
- Without noise, Laser B comes close to the ESN.
- With the laser's own quantum noise, Laser B scores worse than the linear memory reference.
- Averaging many runs or many particles barely helps: 0.736 → 0.673 with 256 averages.

## Files

| Path | Contents |
|---|---|
| `laser_reservoir.ipynb` | The full study, Sections 0–6. Runs in about 1.5 minutes. |
| `choices.md` | Every non-obvious modelling choice, with the reason. |
| `scripts/noise_search.py` | Settings search with quantum noise (Section 5). About 5 minutes. |
| `scripts/averaging_test.py` | Test of averaging repeated runs or many particles. |
| `results/*.json` | Saved results of the longer searches. |
| `visualization/` | A browser diagram that runs the laser reservoir live. |

## How to run

Requirements: Python 3 with `numpy`, `scipy`, `matplotlib` and `numba`.

**Notebook.** Open `laser_reservoir.ipynb` in Jupyter, or in [Google Colab](https://colab.research.google.com/github/lakshsarda137/laser-reservoir-computing/blob/main/laser_reservoir.ipynb). The notebook runs with no other files. Sections 4 and 5 use saved search results. To recompute Section 4, set `RECOMPUTE_SWEEP = True` (about 8 extra minutes).

**Scripts.** Run from the project folder:

```
python3 scripts/noise_search.py
python3 scripts/averaging_test.py
```

**Visualization.**

```
python3 -m http.server 8000 --directory visualization
```

Then open http://localhost:8000/laser_reservoir.html. After editing `template.html` or `core.js`, rebuild with `python3 visualization/build.py`. `node visualization/test_core.js` checks the browser simulation against the notebook.

## References

- Cho, S., Humar, M., Martino, N. & Yun, S. H. Laser particle stimulated emission microscopy. *Physical Review Letters* 117, 193902 (2016).
- Appeltant, L. et al. Information processing using a single dynamical node as complex system. *Nature Communications* 2, 468 (2011).
