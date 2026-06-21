"""WHY does out_scale lower the adaptive loss?
E1 amplitude decomposition, E2 causal add-back, E3 gain/clamp diagnostics.
Forward/cost/trainer all come from the core (fc); only the experiment wiring is here."""
import os
import numpy as np
import torch
import exp_utils as E
import run

fc = E.fc
fc.MODEL_TYPE = "adaptive"
outdir = run.run_dir("adaptive")

np.random.seed(0)
z0 = fc.guess_initial_params_adaptive()
z0_ext = torch.cat([z0, torch.tensor([20.0], dtype=torch.float64, device=E.device)])
bounds = fc.z_bounds_adaptive
bounds_ext = torch.cat([bounds, torch.tensor([[0.1, 200.0]], dtype=torch.float64)], 0)

cost_scale = lambda z, data: fc.model_cost(fc.simulate_adaptive(z[:-1]), data, z[-1])

zA = fc.train_staged(z0, fc.data, fc.calc_cost, bounds, E.LRS, E.STEPS)
zBext = fc.train_staged(z0_ext, fc.data, cost_scale, bounds_ext, E.LRS, E.STEPS)
bestA = fc.calc_cost(zA, fc.data).item()
bestB = cost_scale(zBext, fc.data).item()

s_star, l1, ls = E.best_scale_decomp(fc.simulate_adaptive(zA))           # E1
_, dA = fc.simulate_adaptive(zA, return_diag=True)                       # E3
_, dB = fc.simulate_adaptive(zBext[:-1], return_diag=True)
pA = fc.assign_params_adaptive(zA); gA = (pA["inp_gain"] * pA["out_gain"]).abs()
pB = fc.assign_params_adaptive(zBext[:-1]); gB = (pB["inp_gain"] * pB["out_gain"]).abs()

lines = [
    "WHY out_scale lowers adaptive loss (steps/stage=%d, lrs=%s)" % (E.STEPS, E.LRS),
    "",
    "E2 training best loss:",
    "  no out_scale (270)        = %.3f" % bestA,
    "  + out_scale (271)         = %.3f   out_scale=%.3f" % (bestB, float(zBext[-1])),
    "",
    "E1 amplitude decomposition on trained no-scale model:",
    "  s*=%.4f  loss(s=1)=%.3f  loss(s=s*)=%.3f" % (s_star, l1, ls),
    "  -> s*~1 & loss(s*)~loss(1) means out_scale is NOT a post-hoc rescale.",
    "",
    "E3 diagnostics:",
    "  no-scale  : max|act|=%.3e clamp%%=%.3f gain(inp*out) max=%.2f mean=%.3f"
    % (dA["maxact"], dA["clamp_pct"], float(gA.max()), float(gA.mean())),
    "  +out_scale: max|act|=%.3e clamp%%=%.3f gain(inp*out) max=%.2f mean=%.3f"
    % (dB["maxact"], dB["clamp_pct"], float(gB.max()), float(gB.mean())),
]
print("\n".join(lines))
E.write_lines(os.path.join(outdir, "results.txt"), lines)
run.write_param_table(zA, "adaptive", os.path.join(outdir, "noscale_table.csv"))
run.write_param_table(zBext[:-1], "adaptive", os.path.join(outdir, "outscale_table.csv"),
                      extra_cols={"out_scale": np.full(fc.nofcells, float(zBext[-1]))})
print("saved ->", outdir)
