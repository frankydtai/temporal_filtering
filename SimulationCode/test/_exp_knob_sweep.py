"""WHY do inp_gain and out_scale behave so differently?
From a trained operating point, sweep each knob and record output amplitude,
internal max|activity| and clamp%:
  out_scale = global linear factor OUTSIDE the recurrent loop
  inp_gain  = per-cell gain INSIDE the recurrent ReLU loop
Forward/cost/trainer from the core (fc); results (csv) saved to outdir."""
import os
import numpy as np
import torch
import exp_utils as E
import run

fc = E.fc
fc.MODEL_TYPE = "adaptive"
outdir = run.run_dir("adaptive")

# train an out_scale model (inp_gain fixed = 1) for a realistic, gently-coupled shape
np.random.seed(0)
z0_ext = torch.cat([fc.guess_initial_params_adaptive(),
                    torch.tensor([20.0], dtype=torch.float64, device=E.device)])
bounds_ext = torch.cat([fc.z_bounds_adaptive, torch.tensor([[0.1, 200.0]], dtype=torch.float64)], 0)
cost_os = lambda z, data: fc.model_cost(fc.simulate_adaptive(z[:-1], inp_scalar=1.0), data, z[-1])
z_os_ext = fc.train_staged(z0_ext, fc.data, cost_os, bounds_ext, E.LRS, E.STEPS)
z = z_os_ext[:-1]; s_tr = float(z_os_ext[-1])

m0, d0 = fc.simulate_adaptive(z, inp_scalar=1.0, return_diag=True)   # baseline shape, inp=1
amp0 = float(m0.abs().max())
data_amp = float(fc.data[50:200].abs().max())

rows = ["knob,value,out_amp,max_activity,clamp_pct,loss"]
print("trained out_scale best=%.3f out_scale=%.3f baseline out_amp(inp=1)=%.4f data_amp=%.3f"
      % (cost_os(z_os_ext, fc.data).item(), s_tr, amp0, data_amp))

# knob 1: out_scale (outside loop) -- amplitude linear, dynamics unchanged
print("\n=== out_scale (OUTSIDE loop) ===\n  value   out_amp   max|act|  clamp%  loss")
for k in [1, 5, 10, s_tr, 50, 100]:
    loss = float(fc.model_cost(m0, fc.data, k))
    print("  %6.2f  %8.3f  %8.3f  %5.2f  %8.3f" % (k, k * amp0, d0["maxact"], d0["clamp_pct"], loss))
    rows.append("out_scale,%.4f,%.6f,%.6f,%.4f,%.6f" % (k, k * amp0, d0["maxact"], d0["clamp_pct"], loss))

# knob 2: inp_gain (inside loop) -- amplitude nonlinear, dynamics destabilise
print("\n=== inp_gain (INSIDE loop) ===\n  value   out_amp      max|act|   clamp%  loss")
for k in [1, 2, 3, 5, 8, 12, 20]:
    m, dg = fc.simulate_adaptive(z, inp_scalar=k, return_diag=True)
    loss = float(fc.model_cost(m, fc.data))
    print("  %6.2f  %10.4f  %.3e  %5.2f  %.3e" % (k, float(m.abs().max()), dg["maxact"], dg["clamp_pct"], loss))
    rows.append("inp_gain,%.4f,%.6f,%.6e,%.4f,%.6e" % (k, float(m.abs().max()), dg["maxact"], dg["clamp_pct"], loss))

E.write_lines(os.path.join(outdir, "knob_sweep.csv"), rows)
print("\nsaved ->", outdir)
