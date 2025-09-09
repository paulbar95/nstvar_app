# service/validate.py
from __future__ import annotations
import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import xarray as xr
from minio import Minio

CACHE_DIR = "/app/cache"
INDEX_JSON = os.path.join(CACHE_DIR, "pm25_index.json")

from service.country_mask import target_grid, DEFAULT_MASK_PATH

# ---------- kleine Hilfen ----------

def _load_index() -> Dict[str, Any]:
    if not os.path.exists(INDEX_JSON):
        raise FileNotFoundError("Index not found. Run POST /api/pm25/files/reindex first.")
    import json
    with open(INDEX_JSON, "r") as f:
        return json.load(f)

def _coords(ds: xr.Dataset) -> Tuple[str, str]:
    cand_lat = ["lat", "latitude", "nav_lat", "y"]
    cand_lon = ["lon", "longitude", "nav_lon", "x"]
    latn = next((c for c in cand_lat if c in ds.coords), None)
    lonn = next((c for c in cand_lon if c in ds.coords), None)
    if not latn or not lonn:
        raise ValueError("Cannot find lat/lon coordinates")
    return latn, lonn

def _varname(ds: xr.Dataset) -> str:
    if "mmrpm2p5" in ds.data_vars:
        return "mmrpm2p5"
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    for v in ds.data_vars:
        if any(x in ds[v].dims for x in ("lat", "latitude", "y")):
            return v
    return list(ds.data_vars)[0]

def _covers_year(meta: Dict[str, Any], year: int) -> bool:
    s = str(meta.get("start") or "")
    e = str(meta.get("end") or "")
    if len(s) != 6 or len(e) != 6:
        return False
    ys = int(s[:4]); ye = int(e[:4])
    return ys <= year <= ye

def _download_nc(client: Minio, bucket: str, key: str, dest_dir: str = CACHE_DIR) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    local = os.path.join(dest_dir, f"dl_{os.path.basename(key)}")
    if not os.path.exists(local):
        client.fget_object(bucket, key, local)
    return local

# ---------- Kernprüfungen ----------

def mask_info(mask_path: Optional[str] = None) -> Dict[str, Any]:
    path = mask_path or os.getenv("COUNTRY_MASK_NC", DEFAULT_MASK_PATH)
    if not os.path.exists(path):
        return {"exists": False, "path": path}
    ds = xr.open_dataset(path)
    try:
        lat = ds["lat"].values
        lon = ds["lon"].values
        iso2_ok = "iso2" in ds.variables
        return {
            "exists": True,
            "path": path,
            "shape": [int(lat.size), int(lon.size)],
            "lat_first_last": [float(lat[0]), float(lat[-1])],
            "lon_first_last": [float(lon[0]), float(lon[-1])],
            "has_iso2": iso2_ok,
            "attrs": dict(ds.attrs),
        }
    finally:
        ds.close()

def check_mask_vs_target(mask_path: Optional[str] = None, res_deg: float = 1.0, tol: float = 1e-6) -> Dict[str, Any]:
    """Vergleicht Maske mit dem erwarteten Zielgrid (lat/lon arrays)."""
    inf = mask_info(mask_path)
    if not inf.get("exists"):
        return {"ok": False, "reason": "mask_missing", "mask": inf}
    path = inf["path"]
    ds = xr.open_dataset(path)
    try:
        mlat = ds["lat"].values
        mlon = ds["lon"].values
    finally:
        ds.close()
    tlat, tlon = target_grid(res_deg=res_deg)
    same_shape = (mlat.size == tlat.size) and (mlon.size == tlon.size)
    lat_diff = float(np.max(np.abs(mlat - tlat))) if same_shape else None
    lon_diff = float(np.max(np.abs(mlon - tlon))) if same_shape else None
    ok = same_shape and (lat_diff is not None and lat_diff <= tol) and (lon_diff is not None and lon_diff <= tol)
    return {
        "ok": ok,
        "same_shape": same_shape,
        "lat_max_abs_diff": lat_diff,
        "lon_max_abs_diff": lon_diff,
        "mask": inf,
        "target": {"shape": [tlat.size, tlon.size], "lat_first_last": [float(tlat[0]), float(tlat[-1])], "lon_first_last": [float(tlon[0]), float(tlon[-1])]},
        "tol": tol,
    }

def regridded_annual2015_on_target(nc_path: str, res_deg: float = 1.0) -> xr.DataArray:
    """wie in threshold_map: 2015-Jahresmittel -> auf 1° Grid -> lat/lon heißen 'lat','lon'."""
    ds = xr.open_dataset(nc_path)
    try:
        var = _varname(ds)
        latn, lonn = _coords(ds)
        da = ds[var]
        if "time" not in da.dims:
            raise ValueError("variable has no time dimension")
        years = xr.DataArray(da["time"].dt.year, dims=("time",))
        da2015 = da.where(years == 2015, drop=True)
        if da2015.sizes.get("time", 0) == 0:
            raise ValueError("no samples with year==2015")
        da_y = da2015.mean(dim="time", skipna=True)
        for d in list(da_y.dims):
            if d not in {latn, lonn}:
                da_y = da_y.mean(dim=d, skipna=True)
        # lon 0..360
        lon_vals = ds[lonn].values
        if np.nanmin(lon_vals) < 0:
            lon_vals = (lon_vals + 360) % 360
            da_y = da_y.assign_coords({lonn: lon_vals}).sortby(lonn)
        tlat, tlon = target_grid(res_deg=res_deg)
        da_t = da_y.interp(
            {latn: xr.DataArray(tlat, dims=("lat",)),
             lonn: xr.DataArray(tlon, dims=("lon",))},
            method="nearest"
        )
        return da_t.rename({latn: "lat", lonn: "lon"})
    finally:
        ds.close()

def check_alignment_with_sample(
        client: Minio,
        bucket: str,
        scenario: Optional[str] = None,
        mask_path: Optional[str] = None,
        res_deg: float = 1.0,
        tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Nimmt die erste Datei (optional gefiltert nach scenario), die 2015 abdeckt,
    regriddet sie auf das Zielgrid und vergleicht die Koordinaten mit der Maske.
    """
    m = check_mask_vs_target(mask_path=mask_path, res_deg=res_deg, tol=tol)
    if not m["ok"]:
        return {"ok": False, "phase": "mask_vs_target", "details": m}

    idx = _load_index()
    files: List[Dict[str, Any]] = idx.get("files", [])
    cand = [f for f in files if _covers_year(f, 2015)]
    if scenario:
        cand = [f for f in cand if str(f.get("scenario","")).lower() == scenario.lower()]
    if not cand:
        return {"ok": False, "phase": "pick_file", "reason": "no_candidate_covering_2015", "scenario": scenario}

    pick = cand[0]
    key = pick.get("key") or pick.get("Key") or pick.get("name")
    local = _download_nc(client, bucket, key)

    da = regridded_annual2015_on_target(local, res_deg=res_deg)
    tlat = da["lat"].values
    tlon = da["lon"].values

    # Maske laden
    ds_mask = xr.open_dataset(m["mask"]["path"])
    try:
        mlat = ds_mask["lat"].values
        mlon = ds_mask["lon"].values
    finally:
        ds_mask.close()

    same_shape = (mlat.size == tlat.size) and (mlon.size == tlon.size)
    lat_diff = float(np.max(np.abs(mlat - tlat))) if same_shape else None
    lon_diff = float(np.max(np.abs(mlon - tlon))) if same_shape else None
    ok = same_shape and (lat_diff is not None and lat_diff <= tol) and (lon_diff is not None and lon_diff <= tol)

    return {
        "ok": ok,
        "phase": "model_vs_mask",
        "file": {"key": key, "model": pick.get("model"), "scenario": pick.get("scenario")},
        "same_shape": same_shape,
        "lat_max_abs_diff": lat_diff,
        "lon_max_abs_diff": lon_diff,
        "mask_shape": [int(mlat.size), int(mlon.size)],
        "regrid_shape": [int(tlat.size), int(tlon.size)],
        "tol": tol,
    }
