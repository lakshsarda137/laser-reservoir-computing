// Laser reservoir core: NARMA10, rate equations, virtual nodes, ridge readout.
// Same model as laser_reservoir.ipynb, in normalized variables:
//   n = N1 / N_th (carriers),  L = beta*q (output),  p = P / P_th (pump), time in units of tau.
//   dn/dt = p - n(1 + L)
//   dL/dt = (n(L + beta) - L) / tau_c
const Core = (() => {
  const BETA = 1e-3;
  const SPLIT = { washout: 100, train: 900, val: 200, test: 400 };
  const K = SPLIT.washout + SPLIT.train + SPLIT.val + SPLIT.test;
  const RIDGES = [1e-8, 1e-6, 1e-4, 1e-2, 1];

  function rng(seed) {
    // mulberry32
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function narma10(nSteps, seed) {
    const r = rng(seed);
    const u = new Float64Array(nSteps + 1), y = new Float64Array(nSteps + 1);
    for (let k = 0; k <= nSteps; k++) u[k] = 0.5 * r();
    for (let k = 9; k < nSteps; k++) {
      let s = 0;
      for (let i = k - 9; i <= k; i++) s += y[i];
      y[k + 1] = 0.3 * y[k] + 0.05 * y[k] * s + 1.5 * u[k - 9] * u[k] + 0.1;
    }
    // target at step k is y(k+1)
    return { u: u.subarray(0, nSteps), d: y.subarray(1, nSteps + 1) };
  }

  function makeMask(n, seed) {
    const r = rng(seed), m = new Float64Array(n);
    for (let i = 0; i < n; i++) m[i] = 2 * r() - 1;
    return m;
  }

  function solverStep(tauC) {
    return tauC < 0.005 ? 0.001 : 0.005;
  }

  // Drive the laser with the masked input. Returns the recording at the end of every slot,
  // plus a few samples per slot of pump and output for plotting.
  function simulate(cfg, u, mask) {
    const N = mask.length, dt = solverStep(cfg.tauC);
    const sps = Math.max(1, Math.round(cfg.theta / dt));
    const S = Math.min(4, sps);                       // plot samples per slot
    const sampleAt = new Uint8Array(sps);
    for (let j = 0; j < S; j++) sampleAt[Math.floor(((j + 1) * sps) / S) - 1] = 1;
    const delay = N * sps;
    const hist = new Float64Array(delay);
    const states = new Float64Array(u.length * N);
    const traceL = new Float32Array(u.length * N * S), traceP = new Float32Array(u.length * N * S);
    const eta = cfg.loop ? cfg.eta : 0, tc = cfg.tauC, b = BETA;
    let n = 0, L = 0, j = 0, t = 0;
    for (let k = 0; k < u.length; k++) {
      for (let i = 0; i < N; i++) {
        const drive = cfg.pBias + cfg.gamma * mask[i] * u[k];
        for (let s = 0; s < sps; s++) {
          let p = drive + eta * hist[j];
          if (p < 0) p = 0;
          const a1 = p - n * (1 + L), c1 = (n * (L + b) - L) / tc;
          let n2 = n + 0.5 * dt * a1, L2 = L + 0.5 * dt * c1;
          const a2 = p - n2 * (1 + L2), c2 = (n2 * (L2 + b) - L2) / tc;
          n2 = n + 0.5 * dt * a2; L2 = L + 0.5 * dt * c2;
          const a3 = p - n2 * (1 + L2), c3 = (n2 * (L2 + b) - L2) / tc;
          n2 = n + dt * a3; L2 = L + dt * c3;
          const a4 = p - n2 * (1 + L2), c4 = (n2 * (L2 + b) - L2) / tc;
          n += (dt / 6) * (a1 + 2 * a2 + 2 * a3 + a4);
          L += (dt / 6) * (c1 + 2 * c2 + 2 * c3 + c4);
          hist[j] = L;
          j = j + 1 === delay ? 0 : j + 1;
          if (sampleAt[s]) { traceL[t] = L; traceP[t] = p; t++; }
        }
        states[k * N + i] = L;
      }
    }
    return { states, traceL, traceP, S, N };
  }

  // Solve (A + ridge*I) w = rhs for symmetric positive definite A (Cholesky). A is d*d, row-major.
  function ridgeSolve(A, rhs, d, ridge) {
    const Lm = new Float64Array(d * d);
    for (let i = 0; i < d; i++) {
      for (let k = 0; k <= i; k++) {
        let s = A[i * d + k] + (i === k ? ridge : 0);
        for (let m = 0; m < k; m++) s -= Lm[i * d + m] * Lm[k * d + m];
        Lm[i * d + k] = i === k ? Math.sqrt(Math.max(s, 1e-300)) : s / Lm[k * d + k];
      }
    }
    const z = new Float64Array(d), w = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      let s = rhs[i];
      for (let m = 0; m < i; m++) s -= Lm[i * d + m] * z[m];
      z[i] = s / Lm[i * d + i];
    }
    for (let i = d - 1; i >= 0; i--) {
      let s = z[i];
      for (let m = i + 1; m < d; m++) s -= Lm[m * d + i] * w[m];
      w[i] = s / Lm[i * d + i];
    }
    return w;
  }

  // Features: recordings rescaled with training statistics, then optionally [u(k), 1]
  function buildFeatures(states, N, u, withInput) {
    const k0 = SPLIT.washout, k1 = k0 + SPLIT.train;
    const mean = new Float64Array(N), sd = new Float64Array(N);
    for (let k = k0; k < k1; k++) for (let i = 0; i < N; i++) mean[i] += states[k * N + i];
    for (let i = 0; i < N; i++) mean[i] /= SPLIT.train;
    for (let k = k0; k < k1; k++) for (let i = 0; i < N; i++) { const v = states[k * N + i] - mean[i]; sd[i] += v * v; }
    for (let i = 0; i < N; i++) sd[i] = Math.sqrt(sd[i] / SPLIT.train) + 1e-12;
    const d = N + (withInput ? 2 : 1);
    const F = new Float64Array(u.length * d);
    for (let k = 0; k < u.length; k++) {
      for (let i = 0; i < N; i++) F[k * d + i] = (states[k * N + i] - mean[i]) / sd[i];
      if (withInput) { F[k * d + N] = u[k]; F[k * d + N + 1] = 1; } else F[k * d + N] = 1;
    }
    return { F, d };
  }

  function gram(F, d, k0, k1) {
    const A = new Float64Array(d * d);
    for (let k = k0; k < k1; k++) {
      const off = k * d;
      for (let i = 0; i < d; i++) {
        const fi = F[off + i];
        if (fi === 0) continue;
        for (let m = 0; m <= i; m++) A[i * d + m] += fi * F[off + m];
      }
    }
    for (let i = 0; i < d; i++) for (let m = 0; m < i; m++) A[m * d + i] = A[i * d + m];
    return A;
  }

  function project(F, d, target, k0, k1) {
    const r = new Float64Array(d);
    for (let k = k0; k < k1; k++) for (let i = 0; i < d; i++) r[i] += F[k * d + i] * target[k];
    return r;
  }

  function predict(F, d, w, k) {
    let s = 0;
    for (let i = 0; i < d; i++) s += F[k * d + i] * w[i];
    return s;
  }

  function nmse(F, d, w, target, k0, k1) {
    let mean = 0;
    for (let k = k0; k < k1; k++) mean += target[k];
    mean /= k1 - k0;
    let err = 0, v = 0;
    for (let k = k0; k < k1; k++) {
      const e = predict(F, d, w, k) - target[k];
      err += e * e;
      v += (target[k] - mean) ** 2;
    }
    return err / v;
  }

  // Train the readout like the notebook: ridge chosen on validation, refit on train + validation.
  function trainReadout(states, N, u, target) {
    const { F, d } = buildFeatures(states, N, u, true);
    const tr0 = SPLIT.washout, tr1 = tr0 + SPLIT.train, va1 = tr1 + SPLIT.val, te1 = va1 + SPLIT.test;
    const Atr = gram(F, d, tr0, tr1), btr = project(F, d, target, tr0, tr1);
    let best = RIDGES[0], bestScore = Infinity;
    for (const r of RIDGES) {
      const s = nmse(F, d, ridgeSolve(Atr, btr, d, r), target, tr1, va1);
      if (s < bestScore) { bestScore = s; best = r; }
    }
    const w = ridgeSolve(gram(F, d, tr0, va1), project(F, d, target, tr0, va1), d, best);
    const pred = new Float64Array(u.length);
    for (let k = 0; k < u.length; k++) pred[k] = predict(F, d, w, k);
    return { w, d, F, pred, ridge: best, testNmse: nmse(F, d, w, target, va1, te1), valNmse: bestScore };
  }

  // Memory and nonlinearity tests (notebook Section 3.3): recover u(k-j) and the product of
  // mean-removed inputs u(k)u(k-j) from the recordings alone. Returns R² on validation.
  function memoryCurves(states, N, u, maxLag) {
    const { F, d } = buildFeatures(states, N, u, false);
    const tr0 = SPLIT.washout, tr1 = tr0 + SPLIT.train, va1 = tr1 + SPLIT.val;
    const A = gram(F, d, tr0, tr1);
    let um = 0;
    for (let k = tr0; k < tr1; k++) um += u[k];
    um /= SPLIT.train;
    const memory = [], product = [];
    const t = new Float64Array(u.length);
    for (let lag = 0; lag <= maxLag; lag++) {
      for (let k = 0; k < u.length; k++) t[k] = k - lag >= 0 ? u[k - lag] : 0;
      memory.push(1 - nmse(F, d, ridgeSolve(A, project(F, d, t, tr0, tr1), d, 1e-6), t, tr1, va1));
      if (lag >= 1) {
        for (let k = 0; k < u.length; k++) t[k] = k - lag >= 0 ? (u[k] - um) * (u[k - lag] - um) : 0;
        product.push(1 - nmse(F, d, ridgeSolve(A, project(F, d, t, tr0, tr1), d, 1e-6), t, tr1, va1));
      }
    }
    return { memory, product };
  }

  return { BETA, SPLIT, K, narma10, makeMask, simulate, trainReadout, memoryCurves, solverStep };
})();
if (typeof module !== "undefined") module.exports = Core;
