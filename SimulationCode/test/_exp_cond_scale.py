"""Does a global out_scale help the CONDUCTANCE model? (Vm bounded by reversal
potentials, yet out_scale still helps -- same amplitude/shape decoupling as adaptive.)
  baseline : conductance as-is (138 params)
  +scale   : conductance + 1 global out_scale (139 params)
Forward/cost/trainer from the core (fc); results + per-neuron table saved to outdir."""
import os
import numpy as np
import torch
import exp_utils as E
import run

fc = E.fc
fc.MODEL_TYPE = "conductance"
outdir = run.run_dir("conductance")
LRS = (0.1, 0.01, 0.001)          # conductance's native schedule

np.random.seed(0)
z0 = fc.guess_initial_params()
z0_ext = torch.cat([z0, torch.tensor([1.0], dtype=torch.float64, device=E.device)])
bounds = fc.z_bounds
bounds_ext = torch.cat([bounds, torch.tensor([[0.1, 200.0]], dtype=torch.float64)], 0)

cost_scale = lambda z, data: fc.model_cost(fc.simulate_conductance(z[:-1]), data, z[-1])

zA = fc.train_staged(z0, fc.data, fc.calc_cost, bounds, LRS, E.STEPS)
zBext = fc.train_staged(z0_ext, fc.data, cost_scale, bounds_ext, LRS, E.STEPS)
bestA = fc.calc_cost(zA, fc.data).item()
bestB = cost_scale(zBext, fc.data).item()

s_star, l1, ls = E.best_scale_decomp(fc.simulate_conductance(zA))
rawA = float(fc.simulate_conductance(zA).abs().max())
rawB = float(fc.simulate_conductance(zBext[:-1]).abs().max())
gA = (fc.calc_multi_col_params(zA[0:65]) * fc.calc_multi_col_params(zA[65:130])).abs()
gB = (fc.calc_multi_col_params(zBext[0:65]) * fc.calc_multi_col_params(zBext[65:130])).abs()

lines = [
    "conductance + out_scale (steps/stage=%d, lrs=%s)" % (E.STEPS, LRS),
    "  data amplitude max|data| = %.3f" % float(fc.data[50:200].abs().max()),
    "",
    "  baseline (138)            best=%.3f" % bestA,
    "  + out_scale (139)         best=%.3f  out_scale=%.3f" % (bestB, float(zBext[-1])),
    "",
    "amplitude decomposition on trained baseline:",
    "  raw max|model|=%.3f  s*=%.4f  loss(s=1)=%.3f  loss(s=s*)=%.3f" % (rawA, s_star, l1, ls),
    "  -> out_scale is NOT a post-hoc amplitude fix.",
    "",
    "regime diagnostics:",
    "  baseline : raw max|model|=%7.3f  gain(inp*out) max=%.2f mean=%.3f" % (rawA, float(gA.max()), float(gA.mean())),
    "  +out_scale: raw max|model|=%7.3f x out_scale %.2f = %7.3f  gain max=%.2f mean=%.3f"
    % (rawB, float(zBext[-1]), rawB * float(zBext[-1]), float(gB.max()), float(gB.mean())),
]
print("\n".join(lines))
E.write_lines(os.path.join(outdir, "results.txt"), lines)
run.write_param_table(zA, "conductance", os.path.join(outdir, "baseline_table.csv"))
run.write_param_table(zBext[:-1], "conductance", os.path.join(outdir, "outscale_table.csv"),
                      extra_cols={"out_scale": np.full(fc.nofcells, float(zBext[-1]))})
print("saved ->", outdir)
