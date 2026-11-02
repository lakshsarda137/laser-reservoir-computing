"""Laser reservoir with the laser's own quantum noise: re-tune drive settings and score on NARMA10.

Writes results/noise_search.json. Run from the project folder:  python3 scripts/noise_search.py
Uses the same data, split, masks and readout as laser_reservoir.ipynb (Sections 0 and 3).
Takes about 5 minutes with 8 processes.
"""
import os

for var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = "1"

import itertools
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numba import njit

DT = 0.0025          # Euler-Maruyama step (tau_c / 4); without noise it reproduces the RK4 score
TAU_C = 0.01
BETAS = [1e-2, 1e-3, 1e-4, 1e-5]
RIDGES = [1e-8, 1e-6, 1e-4, 1e-2, 1]
GRIDS = {
    "A": dict(p_bias=[0.8, 1.0, 1.3, 2.0], gamma=[0.5, 2, 4, 8], theta=[0.005, 0.01, 0.02], eta=[0.0]),
    "B": dict(p_bias=[1.1, 1.3, 1.6, 2.0, 3.0], gamma=[0.05, 0.1, 0.2, 0.5, 1.0], theta=[0.2, 0.5], eta=[0.3, 0.6, 0.9]),
}


@njit(cache=True)
def run_noisy_laser(u, mask, p_bias, gamma, theta, eta, dt, beta, tau_c, noise, seed):
    """Same reservoir as the notebook, in raw numbers (N1 carriers, q photons, tau = 1), with noise.

    Each process happens at random times (Poisson statistics), so in a step dt its count has
    variance rate * dt: pump, stimulated emission and spontaneous emission into the mode
    (each moves one carrier into one photon), other spontaneous emission, and photon loss.
    """
    np.random.seed(seed)
    n_nodes = mask.size
    steps_per_slot = int(round(theta / dt))
    delay_steps = n_nodes * steps_per_slot
    p_th = 1.0 / (beta * tau_c)
    history = np.zeros(delay_steps)
    states = np.empty((u.size, n_nodes))
    N, q = 0.0, 0.0
    j = 0
    sqdt = np.sqrt(dt)
    for k in range(u.size):
        for i in range(n_nodes):
            for _ in range(steps_per_slot):
                p = p_bias + gamma * mask[i] * u[k] + eta * history[j]
                P = max(p, 0.0) * p_th
                into_mode = beta * N * q + beta * N      # stimulated + spontaneous into the mode
                other = (1.0 - beta) * N                 # spontaneous emission into other modes
                loss = q / tau_c
                dN = (P - into_mode - other) * dt
                dq = (into_mode - loss) * dt
                if noise:
                    x = np.sqrt(into_mode) * sqdt * np.random.randn()
                    dN += -x + np.sqrt(P + other) * sqdt * np.random.randn()
                    dq += x - np.sqrt(loss) * sqdt * np.random.randn()
                N = max(N + dN, 0.0)
                q = max(q + dq, 0.0)
                history[j] = beta * q
                j = (j + 1) % delay_steps
            states[k, i] = beta * q
    return states


def narma10(n_steps, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n_steps)
    y = np.zeros(n_steps)
    for k in range(9, n_steps - 1):
        y[k + 1] = 0.3 * y[k] + 0.05 * y[k] * np.sum(y[k - 9:k + 1]) + 1.5 * u[k - 9] * u[k] + 0.1
    return u, y


# Same data and split as the notebook
WASHOUT, N_TRAIN, N_VAL, N_TEST = 200, 3000, 800, 1000
_u, _y = narma10(WASHOUT + N_TRAIN + N_VAL + N_TEST + 1, 42)
U, D = _u[:-1], _y[1:]
TRAIN = slice(WASHOUT, WASHOUT + N_TRAIN)
VAL = slice(WASHOUT + N_TRAIN, WASHOUT + N_TRAIN + N_VAL)
TEST = slice(WASHOUT + N_TRAIN + N_VAL, len(U))
TRAIN_VAL = slice(WASHOUT, WASHOUT + N_TRAIN + N_VAL)


def make_mask(seed, n_nodes=100):
    return np.random.default_rng(seed).uniform(-1, 1, n_nodes)


def nmse(prediction, target):
    return np.mean((prediction - target) ** 2) / np.var(target)


def features(states):
    mean, std = states[TRAIN].mean(0), states[TRAIN].std(0) + 1e-12
    return np.column_stack([(states - mean) / std, U, np.ones(len(U))])


def train_readout(F, target, ridge):
    return np.linalg.solve(F.T @ F + ridge * np.eye(F.shape[1]), F.T @ target)


def best_ridge(F):
    return min(RIDGES, key=lambda r: nmse(F[VAL] @ train_readout(F[TRAIN], D[TRAIN], r), D[VAL]))


def run(beta, s, mask_seed, noise=True):
    return run_noisy_laser(U, make_mask(mask_seed), s["p_bias"], s["gamma"], s["theta"], s["eta"],
                           DT, beta, TAU_C, noise, mask_seed)


def validation_job(job):
    beta, version, s = job
    states = run(beta, s, 0)
    if not np.all(np.isfinite(states)):
        return job, np.inf
    F = features(states)
    return job, min(nmse(F[VAL] @ train_readout(F[TRAIN], D[TRAIN], r), D[VAL]) for r in RIDGES)


def test_job(job):
    beta, version, s, mask_seed, noise = job
    states = run(beta, s, mask_seed, noise)
    F = features(states)
    W = train_readout(F[TRAIN_VAL], D[TRAIN_VAL], best_ridge(F))
    return job, nmse(F[TEST] @ W, D[TEST]), float(states[TRAIN].mean() / beta)


if __name__ == "__main__":
    start = time.time()
    jobs = [(beta, version, dict(zip(grid, values)))
            for beta in BETAS for version, grid in GRIDS.items() for values in itertools.product(*grid.values())]
    with ProcessPoolExecutor(8) as pool:
        val_results = list(pool.map(validation_job, jobs, chunksize=2))
        print(f"validation search done ({time.time() - start:.0f} s)", flush=True)
        best = {}
        for (beta, version, s), score in val_results:
            if score < best.get((beta, version), (np.inf,))[0]:
                best[(beta, version)] = (score, s)
        # Test: 5 masks with noise; also the same settings without noise, for comparison
        test_jobs = [(beta, version, s, seed, noise) for (beta, version), (_, s) in best.items()
                     for seed in range(5) for noise in (True, False)]
        test_results = list(pool.map(test_job, test_jobs))
    out = {}
    for (beta, version, s, seed, noise), score, photons in test_results:
        entry = out.setdefault(str(beta), {}).setdefault(version, dict(
            settings=s, val_nmse=float(best[(beta, version)][0]), test_nmse=[], test_nmse_without_noise=[], mean_photons=photons))
        entry["test_nmse" if noise else "test_nmse_without_noise"].append(float(score))
    path = pathlib.Path(__file__).parent.parent / "results" / "noise_search.json"
    path.write_text(json.dumps(out, indent=1))
    for beta in BETAS:
        for version in GRIDS:
            e = out[str(beta)][version]
            print(f"beta={beta:g} {version}: test NMSE with noise {np.mean(e['test_nmse']):.3f} ± {np.std(e['test_nmse']):.3f}, "
                  f"same settings without noise {np.mean(e['test_nmse_without_noise']):.3f}, "
                  f"photons ≈ {e['mean_photons']:.0f}, settings {e['settings']}")
    print(f"total {time.time() - start:.0f} s")
