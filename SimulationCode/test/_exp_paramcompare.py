"""Head-to-head: which 2nd amplitude knob is better, given per-cell out_gain?
  A: out_gain + out_scale (1 global), inp_gain fixed = 1   (knob outside the loop)
  B: out_gain + inp_gain (per-cell), no out_scale          (= current model)
Forward/cost/trainer from the core (fc); only the experiment wiring is here."""
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
boundsB = fc.z_bounds_adaptive
boundsA = torch.cat([boundsB, torch.tensor([[0.1, 200.0]], dtype=torch.float64)], 0)

cost_A = lambda z, data: fc.model_cost(fc.simulate_adaptive(z[:-1], inp_scalar=1.0), data, z[-1])

zB = fc.train_staged(z0, fc.data, fc.calc_cost, boundsB, E.LRS, E.STEPS)          # inp_gain, no scale
zAext = fc.train_staged(z0_ext, fc.data, cost_A, boundsA, E.LRS, E.STEPS)         # out_scale, inp=1
bestB = fc.calc_cost(zB, fc.data).item()
bestA = cost_A(zAext, fc.data).item()

_, dB = fc.simulate_adaptive(zB, return_diag=True)
_, dA = fc.simulate_adaptive(zAext[:-1], inp_scalar=1.0, return_diag=True)
pB = fc.assign_params_adaptive(zB); gB = (pB["inp_gain"] * pB["out_gain"]).abs()
gA = fc.assign_params_adaptive(zAext[:-1])["out_gain"].abs()                      # inp=1

lines = [
    "inp_gain vs out_scale as 2nd knob (steps/stage=%d, lrs=%s)" % (E.STEPS, E.LRS),
    "",
    "  B: out_gain + inp_gain (per-cell, NO scale) #params=%d  best=%.3f" % (fc.nofparams_adaptive, bestB),
    "  A: out_gain + out_scale (1 global, inp=1)   #params=%d  best=%.3f  out_scale=%.3f"
    % (fc.nofparams_adaptive + 1, bestA, float(zAext[-1])),
    "",
    "diagnostics:",
    "  B: max|act|=%.3e gain(inp*out) max=%.2f mean=%.3f" % (dB["maxact"], float(gB.max()), float(gB.mean())),
    "  A: max|act|=%.3e gain(out)     max=%.2f mean=%.3f" % (dA["maxact"], float(gA.max()), float(gA.mean())),
]
print("\n".join(lines))
E.write_lines(os.path.join(outdir, "results.txt"), lines)
run.write_param_table(zB, "adaptive", os.path.join(outdir, "inpgain_table.csv"))
run.write_param_table(zAext[:-1], "adaptive", os.path.join(outdir, "outscale_table.csv"),
                      extra_cols={"out_scale": np.full(fc.nofcells, float(zAext[-1]))})
print("saved ->", outdir)
