# Handoff

Read this first, then `context.md` (the assignment), `explanation_instructions.md` (how to explain things to me), and `choices.md` (every choice so far, with reasons).

## Current state

`laser_reservoir.ipynb` has six sections (0–5), all working. The whole notebook runs in about 1.5 minutes. Section 4 loads its search results from `results/tau_c_sweep.json`; deleting that file makes it recompute them (about 8 extra minutes).

| Notebook section | What it does | Result (test NMSE) |
|---|---|---|
| 0 | NARMA10 data (seed 42), NMSE, split 200 washout / 3000 train / 800 validation / 1000 test | — |
| 1 | ESN baseline, N = 100, tuned on validation | 0.067 |
| 2 | Laser rate equations (Cho et al. PRL 2016), checked against theory | threshold, turn-on delay, relaxation period all match |
| 3 | Laser as reservoir: A = no feedback, B = delay loop, tuned on validation | A: 0.648, B: 0.115. Linear memory reference: 0.326 |
| 4 | Exploration: photon lifetime τ_c = 0.003–0.3 | B: 0.131 → best 0.081 (τ_c = 0.1) → 0.109. A stays ≥ 0.44 |
| 5 | The laser's own quantum noise, re-tuned, β = 1e-2 to 1e-5 | B: 0.61–0.71, A: 0.70–0.77 |

**Main findings:**
- The laser particle alone (A) fails NARMA10. Its memory lasts about 2 input steps; NARMA10 needs 9.
- With a delay loop (B) and a small input strength (γ = 0.05), the laser comes close to the ESN (0.115 vs 0.067). It beats the memory-only level, so its nonlinearity helps.
- τ_c is a weak lever: a 100-fold change gives about a 1.6-fold change in score, with a middle optimum. The input strength γ matters far more.

**History worth knowing (to explain if asked):** the first Section 3 grid used γ ≥ 0.5 and gave B = 0.46. That led to a wrong early conclusion that memory was B's main limit. Widening the grid to small γ fixed it.

**Visualization:** `visualization/` holds one diagram that runs the laser reservoir live in the browser: input → reservoir (pump, laser, 100 recordings, delay loop) → trained weights → sum → prediction, with a loop on/off switch and a prediction-vs-truth chart. Pick a step by clicking the chart, with ← →, or with Play. Edit `template.html` or `core.js`, then run `python3 visualization/build.py` to rebuild `laser_reservoir.html`. View it on localhost: `python3 -m http.server 8000 --directory visualization`, then open http://localhost:8000/laser_reservoir.html. Do not publish it as a claude.ai artifact. `node visualization/test_core.js` checks the browser simulation against the notebook (Laser B, N = 100: test NMSE ≈ 0.13).

## Noise (Section 5, done)

With the laser's own quantum noise, the laser reservoir loses its advantage: Laser B scores 0.67 at β = 1e-3 and at best 0.61 with about 15,000 photons, worse than the linear memory reference (0.33). Near threshold the noise in each recording is about 20× the change the inputs cause. The search is in `scripts/noise_search.py` (about 5 minutes), results in `results/noise_search.json`.

## Other open points

- At τ_c = 0.3, several best settings sit at the grid edge.
- With γ ≥ 0.5, strong feedback hurt. With γ = 0.05, strong feedback is best at large τ_c. Not fully explained.
- The ESN is 2× worse than Appeltant's traditional RC (0.067 vs 0.032 NMSE). Not resolved.

## Remaining deliverables (from `context.md`)

- Slides: done. `slides/laser_reservoir_slides.pptx` (5 slides, speaker notes included). The `slides/` folder is git-ignored, so it exists only on this computer. Rebuild with `npm install --prefix slides` (once), then `node slides/build_slides.js`. Charts read `results/*.json`.
- Check that the notebook runs in Colab.
- Commit and push (only when I ask).

## Step names

`context.md` calls the parameter study "Step 3: one additional exploration question". The notebook calls it **Section 4**. They are the same thing.

## Practical warnings

- **Keep the one-thread setting** at the top of the notebook (`OPENBLAS_NUM_THREADS=1` etc.). Without it, the notebook was about 20× slower.
- **Test speed first.** Time one run before any sweep. Tell me the expected total time before starting anything longer than about a minute.
- **The solver step must scale with τ_c.** `solver_step(tau_c)` in the notebook gives 0.001 for τ_c < 0.005, else 0.005. θ must be a whole multiple of the step. Check any new step by halving it once.
- **Run the notebook** with `jupyter nbconvert --to notebook --execute --inplace laser_reservoir.ipynb` from the project folder (jupyter is in `~/Library/Python/3.11/bin`). Look at the figures before reporting results.
- The laser solver uses `numba` (installed locally and in Colab).
- If the best setting is on a grid edge, say so, and widen the grid or explain why not.

## How I like to work

- Short replies. Follow `explanation_instructions.md` when explaining. Define every technical word.
- Add every non-obvious choice to `choices.md`, with the reason.
- If you find you said something wrong earlier, say so plainly.
- Commit and push only when I ask. The repo is `lakshsarda137/laser-reservoir-computing` (private). The PDFs, `context.md` and `explanation_instructions.md` are gitignored.
