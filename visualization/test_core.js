// Check the browser simulation against the notebook. Run: node visualization/test_core.js
const C = require(require("path").join(__dirname, "core.js"));
const { u, d } = C.narma10(C.K, 42);
const cfgs = {
  "A best": { pBias: 0.8, gamma: 2, theta: 0.005, eta: 0, loop: false, tauC: 0.01 },
  "B best": { pBias: 1.1, gamma: 0.05, theta: 0.2, eta: 0.3, loop: true, tauC: 0.01 },
  "B strong input": { pBias: 1.1, gamma: 0.5, theta: 0.2, eta: 0.3, loop: true, tauC: 0.01 },
  "B tc=0.003": { pBias: 1.1, gamma: 0.05, theta: 0.2, eta: 0.3, loop: true, tauC: 0.003 },
};
for (const N of [50, 100]) for (const [name, cfg] of Object.entries(cfgs)) {
  const t0 = Date.now();
  const sim = C.simulate(cfg, u, C.makeMask(N, 1));
  const t1 = Date.now();
  const r = C.trainReadout(sim.states, N, u, d);
  const m = C.memoryCurves(sim.states, N, u, 12);
  console.log(`N=${N} ${name}: test NMSE ${r.testNmse.toFixed(3)} (sim ${t1 - t0} ms, train ${Date.now() - t1} ms) mem@9 ${m.memory[9].toFixed(2)} prod@9 ${m.product[8].toFixed(2)}`);
}
