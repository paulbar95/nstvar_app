# service/window_field.py
import os
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import xarray as xr
from minio import Minio

CACHE_DIR = "/app/cache"
INDEX_JSON = os.path.join(CACHE_DIR, "pm25_index.json")

def _load_index() -> Dict[str, Any]:
    if not os.path.exists(INDEX_JSON):
        raise FileNotFoundError("Index not found. Run POST /api/pm25/files/reindex first.")
    with open(INDEX_JSON, "r") as f:
        return json.load(f)

def _target_grid() -> Tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-89.5, 89.5, 180)
    lon = np.linspace(0.5, 359.5, 360)
    return lat, lon

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
        dims = set(ds[v].dims)
        if "lat" in dims or "latitude" in dims:
            return v
    return list(ds.data_vars)[0]

def _covers_any(meta: Dict[str, Any], y0: int, y1: int) -> bool:
    s = str(meta.get("start") or "")
    e = str(meta.get("end") or "")
    if len(s) != 6 or len(e) != 6:
        return False
    ys = int(s[:4]); ye = int(e[:4])
    return not (ye < y0 or ys > y1)

def _download_nc(client: Minio, bucket: str, key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, f"dl_{os.path.basename(key)}")
    if not os.path.exists(local):
        client.fget_object(bucket, key, local)
    return local

def _annual_mean_window_on_target(nc_path: str, y0: int, y1: int) -> xr.DataArray:
    ds = xr.open_dataset(nc_path)
    try:
        var = _varname(ds); latn, lonn = _coords(ds)
        da = ds[var]
        if "time" not in da.dims:
            raise ValueError("variable has no time dimension")
        years = xr.DataArray(da["time"].dt.year, dims=("time",))
        sel = da.where((years >= y0) & (years <= y1), drop=True)
        if sel.sizes.get("time", 0) == 0:
            raise ValueError("no samples in window")

        # erst Monats- → Jahresmittel, dann Jahresmittel über Fenster
        # (vereinfacht: direkt über time mitteln ist ok, falls gleichgewichtet)
        da_win = sel.mean(dim="time", skipna=True)

        # alle non-spatial dims wegmitteln
        for dim in list(da_win.dims):
            if dim not in {latn, lonn}:
                da_win = da_win.mean(dim=dim, skipna=True)

        # auf Zielgrid
        tg_lat, tg_lon = _target_grid()
        lon_vals = ds[lonn].values
        if np.nanmin(lon_vals) < 0:
            lon_vals = (lon_vals + 360) % 360
            da_win = da_win.assign_coords({lonn: lon_vals}).sortby(lonn)

        da_t = da_win.interp(
            {latn: xr.DataArray(tg_lat, dims=("lat",)),
             lonn: xr.DataArray(tg_lon, dims=("lon",))},
            method="nearest"
        )
        return da_t.rename({latn: "lat", lonn: "lon"})
    finally:
        ds.close()

def window_field(client: Minio, bucket: str, scenario: str, y0: int, y1: int) -> xr.DataArray:
    idx = _load_index()
    files = idx.get("files", [])
    cands = [f for f in files if (f.get("scenario","").lower()==scenario.lower()) and _covers_any(f, y0, y1)]
    if not cands:
        raise ValueError(f"No files overlapping {y0}-{y1} for {scenario}")

    fields = []
    for f in cands:
        key = f.get("key") or f.get("Key") or f.get("name")
        if not key:
            continue
        try:
            local = _download_nc(client, bucket, key)
            da = _annual_mean_window_on_target(local, y0, y1)
            if np.isfinite(da.values).any():
                fields.append(da.expand_dims(member=[f.get("model","unknown")]))
        except Exception:
            continue

    if not fields:
        raise ValueError("All fields failed in window_field")
    stack = xr.concat(fields, dim="member", join="outer", fill_value=np.nan)
    return stack.median(dim="member", skipna=True)  # dims: lat, lon
