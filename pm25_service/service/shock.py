# service/shock.py
from typing import Dict, Any
import numpy as np
import xarray as xr
from minio import Minio

from service.country_mask import ensure_country_mask
from service.threshold_map import build_threshold_map
from service.window_field import window_field

def _masked_region_agg(field: xr.DataArray, mask_country: xr.DataArray, agg: str) -> float:
    # field: (lat,lon) ; mask_country: (lat,lon) bool
    vals = field.where(mask_country, np.nan).values
    finite = np.isfinite(vals)
    if not finite.any():
        return float("nan")
    arr = vals[finite]
    if agg == "mean":
        return float(np.nanmean(arr))
    return float(np.nanmedian(arr))  # default median

def compute_region_shock(
        client: Minio,
        bucket: str,
        region_iso2: str,
        scenario: str,
        y0: int, y1: int,
        *,
        mode: str = "baseline",         # "baseline" | "percentile"
        q: float = 0.95,                # wenn mode="percentile"
        basis: str = "ensemble2015",    # baseline/q Basis
        agg: str = "median",            # "median" | "mean" (räumlich)
        stat: str = "ratio",            # "ratio"  (W/B - 1) | "delta" (W - B)
) -> Dict[str, Any]:
    region = region_iso2.upper()

    # 1) Stelle sicher, dass Maske existiert
    mi = ensure_country_mask()  # Auto-Download GeoJSON + erzeuge Maske falls fehlt
    mask_path = mi["mask_path"]

    dsm = xr.open_dataset(mask_path)
    try:
        if region not in [str(c) for c in dsm["country"].values]:
            raise ValueError(f"Region {region} not found in mask")
        mask_country = dsm["mask"].sel(country=region)  # (lat,lon) bool
    finally:
        dsm.close()

    # 2) Feld im Zeitfenster (Ensemble-Median)
    field_win = window_field(client, bucket, scenario, y0, y1)  # (lat,lon)

    # 3) Basiskarte laden/erzeugen
    thr = build_threshold_map(client=client, bucket=bucket, scenario=scenario, quantile=q, basis=basis)
    ds_thr = xr.open_dataset(thr["path"])
    try:
        if mode == "percentile":
            ref_map = ds_thr["q"]            # per-cell q-Karte
        else:
            ref_map = ds_thr["baseline"]     # per-cell 2015-baseline (Median über Modelle)
    finally:
        ds_thr.close()

    # 4) räumliche Aggregation auf Region
    W = _masked_region_agg(field_win, mask_country, agg)
    B = _masked_region_agg(ref_map,  mask_country, agg)

    if not np.isfinite(W) or not np.isfinite(B):
        raise ValueError("Insufficient data for region aggregation")

    if stat == "delta":
        shock = W - B
    else:
        shock = (W / B) - 1.0

    return {
        "region": region,
        "scenario": scenario,
        "window": [y0, y1],
        "mode": mode,
        "basis": basis,
        "agg": agg,
        "stat": stat,
        "q": q if mode == "percentile" else None,
        "W_window": W,
        "B_ref": B,
        "shock": shock,
        "maps": {
            "window_field": "in-memory",
            "threshold_map": thr["path"],
            "mask": mask_path,
        },
    }
