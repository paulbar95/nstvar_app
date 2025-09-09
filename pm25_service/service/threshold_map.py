# service/threshold_map.py
import os
import json
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import xarray as xr
from minio import Minio

CACHE_DIR = "/app/cache"
INDEX_JSON = os.path.join(CACHE_DIR, "pm25_index.json")

# -------------------------
# Hilfsfunktionen
# -------------------------

def _target_grid() -> Tuple[np.ndarray, np.ndarray]:
    # 1°-Grid (anpassbar)
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
    # heuristisch: erste Var, die (lat|latitude) als Dim hat
    for v in ds.data_vars:
        dims = set(ds[v].dims)
        if "lat" in dims or "latitude" in dims:
            return v
    # Fallback
    return list(ds.data_vars)[0]

def _is_scenario(meta: Dict[str, Any], scenario: str) -> bool:
    s = (meta.get("scenario") or meta.get("Scenario") or "").lower()
    return s == scenario.lower()

def _covers_year(meta: Dict[str, Any], year: int) -> bool:
    start = str(meta.get("start") or "")
    end   = str(meta.get("end") or "")
    if len(start) != 6 or len(end) != 6:
        return False
    ys = int(start[:4]); ye = int(end[:4])
    return ys <= year <= ye

def _download_nc(client: Minio, bucket: str, key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, f"dl_{os.path.basename(key)}")
    if not os.path.exists(local):
        client.fget_object(bucket, key, local)
    return local

def _annual_mean_2015_on_target(nc_path: str) -> xr.DataArray:
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

        # 2015-Mittel
        da_y = da2015.mean(dim="time", skipna=True)

        # alle Dims außer lat/lon wegmitteln (z.B. 'lev', 'height' etc.)
        for dim in list(da_y.dims):
            if dim not in {latn, lonn}:
                da_y = da_y.mean(dim=dim, skipna=True)

        # Zielgrid, 0..360 Longitudes vereinheitlichen
        tg_lat, tg_lon = _target_grid()
        lon_vals = ds[lonn].values
        if np.nanmin(lon_vals) < 0:
            lon_vals = (lon_vals + 360) % 360
            da_y = da_y.assign_coords({lonn: lon_vals}).sortby(lonn)

        da_t = da_y.interp(
            {latn: xr.DataArray(tg_lat, dims=("lat",)),
             lonn: xr.DataArray(tg_lon, dims=("lon",))},
            method="nearest"
        )
        return da_t.rename({latn: "lat", lonn: "lon"})
    finally:
        ds.close()

def _load_index() -> Dict[str, Any]:
    if not os.path.exists(INDEX_JSON):
        raise ValueError("Index not found. Run POST /api/pm25/files/reindex first.")
    with open(INDEX_JSON, "r") as f:
        return json.load(f)

# -------------------------
# Hauptfunktion
# -------------------------

def build_threshold_map(
        client: Minio,
        bucket: str,
        scenario: str,
        quantile: float,
        basis: str = "ensemble2015",
) -> Dict[str, Any]:
    """
    Zellweise Threshold-Map:
      basis='ensemble2015':
        - pro Modell 2015-Jahresmittel -> Stack über 'member'
        - baseline = Median(member), q = Quantil(member, q)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    index = _load_index()
    files: List[Dict[str, Any]] = index.get("files", [])

    # Kandidaten für dieses Szenario, die 2015 abdecken
    candidate_files = [f for f in files if _is_scenario(f, scenario) and _covers_year(f, 2015)]
    print(f"[thrmap] scenario={scenario} candidates={len(candidate_files)}")

    fields: List[xr.DataArray] = []
    members: List[str] = []
    models: Set[str] = set()
    errors: List[Dict[str, str]] = []

    if basis == "ensemble2015":
        for f in candidate_files:
            key = f.get("key") or f.get("Key") or f.get("name")
            if not key:
                continue
            try:
                local = _download_nc(client, bucket, key)
                da = _annual_mean_2015_on_target(local)
                if not np.isfinite(da.values).any():
                    raise ValueError("field is all-NaN after preprocessing")
                model = f.get("model") or f.get("Model") or "unknown"
                fields.append(da.expand_dims(member=[model]))
                members.append(model)
                models.add(model)
            except Exception as ex:
                errors.append({"key": key, "error": str(ex)})

        if not fields:
            raise ValueError(f"No 2015 fields for scenario {scenario}; first_errors={errors[:3]}")

        # coords können (selten) differieren -> outer-join + NaN-Fill
        stack = xr.concat(fields, dim="member", join="outer", fill_value=np.nan)

        baseline: xr.DataArray = stack.median(dim="member", skipna=True)
        qmap: xr.DataArray     = stack.quantile(quantile, dim="member", skipna=True)
    else:
        raise ValueError(f"Unknown basis '{basis}'")

    # Ausgabe
    out_path = os.path.join(CACHE_DIR, f"thrmap_{basis}_{scenario}_q{quantile:.2f}.nc")
    ds_out = xr.Dataset(
        {"baseline": baseline.astype("float32"), "q": qmap.astype("float32")},
        coords={"lat": baseline["lat"], "lon": baseline["lon"]},
        attrs={
            "basis": basis,
            "scenario": scenario,
            "quantile": quantile,
            "members": ",".join(sorted(models)),
        },
    )
    if os.path.exists(out_path):
        os.remove(out_path)
    ds_out.to_netcdf(out_path)

    qvals = qmap.values.ravel()
    finite = np.isfinite(qvals)
    stats_q = {
        "min": float(np.nanmin(qvals[finite])) if finite.any() else None,
        "p50": float(np.nanmedian(qvals[finite])) if finite.any() else None,
        "max": float(np.nanmax(qvals[finite])) if finite.any() else None,
    }

    return {
        "path": out_path,
        "cached": False,
        "n_candidates": len(candidate_files),
        "n_inputs": int(stack.sizes["member"]),
        "n_used": int(stack.sizes.get("member", 0)),
        "n_models": len(models),
        "basis": basis,
        "scenario": scenario,
        "quantile": quantile,
        "stats_q": stats_q,
        "errors": errors[:5],
    }
