# -*- coding: utf-8 -*-
"""Gruntman moving-bar training target: fig1_ci traces + per-column ``t_center``.

``build_moving_bar_target`` returns batched moving-bar ``signal``, per-readout
``data`` on a fixed ``COST_WINDOW_STEPS`` grid aligned to ``t_center ± 0.45 s``,
and ``cost_t0`` for :mod:`FiveCol_MedSim_Pytorch` windowed cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from Medulla_Library import T_ON
from network.stimulus import build_moving_bar_signals, cost_photo_columns
from t4_t5_preference import READOUT_SUBTYPES, fig1_key_for_stimulus, normalize_side
from training_config import (
    COST_HALF_WINDOW_STEPS,
    COST_WINDOW_STEPS,
    FIG1_CI_NPZ,
)
from visual_stimulus.moving_bar_stimulus import column_bar_center_step

_TRACE_CACHE: Dict[str, np.ndarray] = {}


@dataclass
class MovingBarTarget:
    signal: torch.Tensor          # (B, T, N)
    data: torch.Tensor            # (n_cost, COST_WINDOW_STEPS)
    power: torch.Tensor           # scalar
    cost_weight: torch.Tensor     # (n_cost,)
    cost_t0: torch.Tensor         # (n_cost,) absolute simulation step
    readout_batch: torch.Tensor   # (n_cost,) long
    readout_unit: torch.Tensor    # (n_cost,) long
    n_batch: int
    maxtime: int
    info: dict


def _fig1_trace_ids(npz_path: Path) -> List[str]:
    with np.load(npz_path) as d:
        return sorted({k.replace("__time_ms", "") for k in d.files if k.endswith("__time_ms")})


def load_fig1_trace(
    trace_id: str,
    npz_path: Path = FIG1_CI_NPZ,
    n_steps: int = COST_WINDOW_STEPS,
    half_window_steps: int = COST_HALF_WINDOW_STEPS,
    deltat_ms: float = 10.0,
) -> np.ndarray:
    """Resample one fig1 trace to ``n_steps`` centred at 0 ms (10 ms spacing).

    Digitized ``time_ms`` in the npz is -450..+450 ms relative to bar centre.
    """
    key = f"{trace_id}|{n_steps}|{half_window_steps}|{deltat_ms}"
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]

    with np.load(npz_path) as d:
        t_key, v_key = f"{trace_id}__time_ms", f"{trace_id}__vm_mv"
        if t_key not in d.files:
            raise KeyError(f"missing trace {trace_id!r} in {npz_path}")
        time_ms = np.asarray(d[t_key], dtype=np.float64)
        vm_mv = np.asarray(d[v_key], dtype=np.float64)

    rel_ms = (np.arange(n_steps, dtype=np.float64) - half_window_steps) * deltat_ms
    trace = np.interp(rel_ms, time_ms, vm_mv, left=vm_mv[0], right=vm_mv[-1])
    _TRACE_CACHE[key] = trace
    return trace


def load_fig1_traces(
    npz_path: Path = FIG1_CI_NPZ,
    n_steps: int = COST_WINDOW_STEPS,
    half_window_steps: int = COST_HALF_WINDOW_STEPS,
    deltat_ms: float = 10.0,
) -> Dict[str, np.ndarray]:
    """All fig1 traces resampled to the per-column training window."""
    return {
        tid: load_fig1_trace(tid, npz_path, n_steps, half_window_steps, deltat_ms)
        for tid in _fig1_trace_ids(npz_path)
    }


from .tiling import unit_type_names


def col2subtype(C, u: int, v: int, subtype: str, names: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit indices of ``subtype`` (e.g. ``T4a``) on column ``(u, v)``."""
    if names is None:
        names = unit_type_names(C)
    return np.where(
        (C.u == int(u)) & (C.v == int(v)) & (names == subtype),
    )[0]


def build_moving_bar_target(
    C,
    device: Optional[str] = None,
    t_on: int = T_ON,
    deltat_ms: float = 10.0,
    fig1_path: Path = FIG1_CI_NPZ,
    use_cache: bool = True,
    center_column: bool = False,
) -> MovingBarTarget:
    """Build moving-bar stimulus + fig1 targets for photo columns × T4/T5 subtypes.

    ``center_column=True`` restricts cost to the hex centre column ``(u,v)=(0,0)``.
    Stimulus still drives all photoreceptor columns.
    """
    device = device or C.device
    side = normalize_side(C.meta.get("side", "right"))

    stim = build_moving_bar_signals(
        C, t_on=t_on, deltat_ms=deltat_ms, device=device, use_cache=use_cache,
        network_json=getattr(C, "source_json", None),
    )
    maxtime = int(stim.info["maxtime"])
    field_deg = stim.info["field_deg"]
    fig1 = load_fig1_traces(fig1_path, deltat_ms=deltat_ms)

    present = [st for st in READOUT_SUBTYPES if st in set(C.type_names)]
    if not present:
        raise ValueError("network has no T4a–d / T5a–d subtypes for moving-bar target")

    type_names = unit_type_names(C)
    cols = cost_photo_columns(C, center_column=center_column)
    center_col = cols[0] if center_column else None

    r_batch, r_unit, r_target, r_weight, r_t0 = [], [], [], [], []
    skipped_orthogonal = 0
    for b, spec in enumerate(stim.specs):
        for col in cols:
            t_center = column_bar_center_step(
                col.x, col.y, spec, field_deg, t_on=t_on, deltat_ms=deltat_ms,
            )
            t0 = t_center - COST_HALF_WINDOW_STEPS
            if t0 < 0 or t_center + COST_HALF_WINDOW_STEPS > maxtime:
                raise ValueError(
                    f"cost window out of range for column ({col.u},{col.v}) "
                    f"spec={spec.name}: t_center={t_center}, maxtime={maxtime}"
                )
            for subtype in present:
                trace_id = fig1_key_for_stimulus(side, subtype, spec)
                if trace_id is None:
                    skipped_orthogonal += 1
                    continue
                if trace_id not in fig1:
                    raise KeyError(f"fig1 trace missing: {trace_id}")
                units = col2subtype(C, col.u, col.v, subtype, type_names)
                if len(units) == 0:
                    continue
                target = fig1[trace_id]
                for uidx in units:
                    r_batch.append(b)
                    r_unit.append(int(uidx))
                    r_target.append(target)
                    r_weight.append(1.0)
                    r_t0.append(t0)

    if not r_batch:
        raise ValueError("no moving-bar cost cells (check subtypes and photo columns)")

    data = torch.tensor(np.asarray(r_target), dtype=torch.float64, device=device)
    cost_weight = torch.tensor(np.asarray(r_weight), dtype=torch.float64, device=device)
    readout_batch = torch.tensor(np.asarray(r_batch), dtype=torch.long, device=device)
    readout_unit = torch.tensor(np.asarray(r_unit), dtype=torch.long, device=device)
    cost_t0 = torch.tensor(np.asarray(r_t0), dtype=torch.long, device=device)

    power = torch.sum(cost_weight[:, None] * data ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=torch.float64, device=device)

    info = {
        **stim.info,
        "n_cost": int(data.shape[0]),
        "n_batch": stim.info["n_batch"],
        "n_cost_columns": len(cols),
        "center_column": bool(center_column),
        "cost_column_uv": (int(center_col.u), int(center_col.v)) if center_col else None,
        "side": side,
        "present_subtypes": present,
        "skipped_orthogonal": skipped_orthogonal,
        "fig1_path": str(fig1_path),
        "cost_window_steps": COST_WINDOW_STEPS,
    }
    return MovingBarTarget(
        signal=stim.signal,
        data=data,
        power=power,
        cost_weight=cost_weight,
        cost_t0=cost_t0,
        readout_batch=readout_batch,
        readout_unit=readout_unit,
        n_batch=stim.info["n_batch"],
        maxtime=maxtime,
        info=info,
    )
