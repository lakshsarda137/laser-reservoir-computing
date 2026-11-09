"""Does averaging reduce the laser's quantum noise enough to help on NARMA10?

Two ways to average M measurements, for Laser B:
- repeats: the same inputs are sent through one laser M times. Each run has its own loop.
- particles: M particles receive the same pump at the same moment. One shared loop adds
  their average light back to the pump, so the loop also carries less noise.

Uses the data, split, masks, solver and readout of scripts/noise_search.py.
Run from the project folder:
    python3 scripts/averaging_test.py check      (one run of each kind, with timing)
    python3 scripts/averaging_test.py ideal      (best case for averaging, about 30 s)
    python3 scripts/averaging_test.py            (full test, writes results/averaging_test.json)
"""
import os

for var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = "1"

import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numba import njit

from noise_search import (D, DT, RIDGES, TAU_C, TEST, TRAIN, TRAIN_VAL, U, VAL, best_ridge, features,
                          make_mask, nmse, run_noisy_laser, train_readout)

BETA = 1e-3
M_VALUES = [1, 4, 16, 64, 256]
SETTINGS = [
    dict(p_bias=1.1, gamma=0.05, theta=0.2, eta=0.3),   # best without noise (Section 3)
    dict(p_bias=1.1, gamma=0.2, theta=0.2, eta=0.3),
    dict(p_bias=1.1, gamma=0.5, theta=0.2, eta=0.3),
]
MASKS = range(3)


@njit(cache=True)
def run_shared_loop(u, mask, p_bias, gamma, theta, eta, dt, beta, tau_c, n_particles, seed):
    """M particles with the same pump. The loop adds back their average light. Return the average light.

    Same noisy equations as run_noisy_laser. Each particle has its own random numbers.
    """
    np.random.seed(seed)
    n_nodes = mask.size
    steps_per_slot = int(round(theta / dt))
    delay_steps = n_nodes * steps_per_slot
    p_th = 1.0 / (beta * tau_c)
    history = np.zeros(delay_steps)
    states = np.empty((u.size, n_nodes))
    N = np.zeros(n_particles)
    q = np.zeros(n_particles)
    j = 0
    sqdt = np.sqrt(dt)
    average = 0.0
    for k in range(u.size):
        for i in range(n_nodes):
            for _ in range(steps_per_slot):
                p = p_bias + gamma * mask[i] * u[k] + eta * history[j]
                P = max(p, 0.0) * p_th
                total = 0.0
                for m in range(n_particles):
                    into_mode = beta * N[m] * q[m] + beta * N[m]
                    other = (1.0 - beta) * N[m]
                    loss = q[m] / tau_c
                    dN = (P - into_mode - other) * dt
                    dq = (into_mode - loss) * dt
                    x = np.sqrt(into_mode) * sqdt * np.random.randn()
                    dN += -x + np.sqrt(P + other) * sqdt * np.random.randn()
                    dq += x - np.sqrt(loss) * sqdt * np.random.randn()
                    N[m] = max(N[m] + dN, 0.0)
                    q[m] = max(q[m] + dq, 0.0)
                    total += q[m]
                average = beta * total / n_particles
                history[j] = average
                j = (j + 1) % delay_steps
            states[k, i] = average
    return states


def run_seed(mask_seed, run):
    """Run 0 uses the mask seed, as in noise_search.py, so M = 1 reproduces the notebook."""
    return mask_seed if run == 0 else 100_000 * (mask_seed + 1) + run


def scores(states):
    """Return the validation NMSE (train fit) and the test NMSE (train + validation fit)."""
    F = features(states)
    ridge = best_ridge(F)
    val = nmse(F[VAL] @ train_readout(F[TRAIN], D[TRAIN], ridge), D[VAL])
    test = nmse(F[TEST] @ train_readout(F[TRAIN_VAL], D[TRAIN_VAL], ridge), D[TEST])
    return float(val), float(test)


def one_run(job):
    s, mask_seed, run = job
    return run_noisy_laser(U, make_mask(mask_seed), s["p_bias"], s["gamma"], s["theta"], s["eta"],
                           DT, BETA, TAU_C, True, run_seed(mask_seed, run))


def repeats_scores(pool, s, mask_seed):
    """Average the runs of one laser. Returns scores for every M, from one set of M_VALUES[-1] runs."""
    total = np.zeros((len(U), 100))
    out = {}
    jobs = [(s, mask_seed, run) for run in range(M_VALUES[-1])]
    for count, states in enumerate(pool.map(one_run, jobs, chunksize=4), start=1):
        total += states
        if count in M_VALUES:
            out[count] = scores(total / count)
    return out


def particles_job(job):
    s, mask_seed, m = job
    states = run_shared_loop(U, make_mask(mask_seed), s["p_bias"], s["gamma"], s["theta"], s["eta"],
                             DT, BETA, TAU_C, m, run_seed(mask_seed, 0))
    return job, scores(states)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        s = SETTINGS[0]
        start = time.time()
        one_run((s, 0, 0))     # first call compiles
        print(f"compile + one run: {time.time() - start:.1f} s", flush=True)
        start = time.time()
        noisy = one_run((s, 0, 0))
        print(f"one run: {time.time() - start:.2f} s", flush=True)
        print(f"M = 1, noise: test NMSE {scores(noisy)[1]:.3f} (notebook: 0.769)")
        clean = run_noisy_laser(U, make_mask(0), s["p_bias"], s["gamma"], s["theta"], s["eta"],
                                DT, BETA, TAU_C, False, 0)
        print(f"no noise: test NMSE {scores(clean)[1]:.3f} (notebook: 0.106)")
        start = time.time()
        shared = run_shared_loop(U, make_mask(0), s["p_bias"], s["gamma"], s["theta"], s["eta"],
                                 DT, BETA, TAU_C, 1, 0)
        print(f"shared loop, M = 1: test NMSE {scores(shared)[1]:.3f}, same as a single run: "
              f"{np.array_equal(shared, noisy)} ({time.time() - start:.1f} s, includes compile)")
        sys.exit()

    if len(sys.argv) > 1 and sys.argv[1] == "ideal":
        # Best case for averaging: noise-free recordings plus random numbers of size noise / sqrt(M).
        # The laser's own motion and the loop stay free of noise. Takes about 30 s.
        s = SETTINGS[0]
        for seed in MASKS:
            args = (U, make_mask(seed), s["p_bias"], s["gamma"], s["theta"], s["eta"], DT, BETA, TAU_C)
            clean, noisy = run_noisy_laser(*args, False, seed), run_noisy_laser(*args, True, seed)
            noise = np.std(noisy[TEST] - clean[TEST])
            rng = np.random.default_rng(seed)
            row = {m: scores(clean + rng.normal(0, noise / np.sqrt(m), clean.shape))[1]
                   for m in [1, 256, 1024, 4096, 16384, 65536, 262144, 1048576]}
            print(f"mask {seed}, noise {noise:.3f}: " + "  ".join(f"M={m}: {v:.3f}" for m, v in row.items()), flush=True)
        sys.exit()

    start = time.time()
    results = {"repeats": {}, "particles": {}}
    with ProcessPoolExecutor(8) as pool:
        for s in SETTINGS:
            key = str(s["gamma"])
            per_mask = [repeats_scores(pool, s, mask_seed) for mask_seed in MASKS]
            results["repeats"][key] = {str(m): {"val": [p[m][0] for p in per_mask], "test": [p[m][1] for p in per_mask]}
                                       for m in M_VALUES}
            print(f"repeats, gamma = {s['gamma']} done ({time.time() - start:.0f} s)", flush=True)
        jobs = [(s, mask_seed, m) for s in SETTINGS for mask_seed in MASKS for m in M_VALUES]
        for (s, mask_seed, m), (val, test) in pool.map(particles_job, jobs):
            entry = results["particles"].setdefault(str(s["gamma"]), {}).setdefault(str(m), {"val": [], "test": []})
            entry["val"].append(val)
            entry["test"].append(test)
        print(f"particles done ({time.time() - start:.0f} s)", flush=True)
    path = pathlib.Path(__file__).parent.parent / "results" / "averaging_test.json"
    path.write_text(json.dumps(results, indent=1))
    for method, by_gamma in results.items():
        print(f"\n{method}: mean test NMSE over {len(MASKS)} masks (validation NMSE in brackets)")
        for key, by_m in by_gamma.items():
            row = "  ".join(f"M={m}: {np.mean(e['test']):.3f} ({np.mean(e['val']):.3f})" for m, e in by_m.items())
            print(f"  gamma={key}:  {row}")
    print(f"\ntotal {time.time() - start:.0f} s")
