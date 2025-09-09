# service/shock.py
import os
import json
from typing import Dict, Any, List, Tuple, Set, Optional, Literal

import numpy as np
import xarray as xr
from minio import Minio

from service.country_mask import DEFAULT_MASK_PATH, mask_info
from service.threshold_map import _load_index as _load_idx_json
from service.threshold_map import _download_nc as _fetch_nc
from service.threshold_map import _coords as _coords
from service.threshold_map import _varname as _varname
from service.threshold_map import _target_grid as _target_grid

CACHE_DIR = "/app/cache"

def _covers_year_range(meta: Dict[str, Any], y0: int, y1: int) -> bool:
    s = str(meta.get("start") or "")
    e = str(meta.get("end") or "")
    if len(s) != 6 or len(e) != 6:
        return False
    ys = int(s[:4]); ye = int(e[:4])
    return not (ye < y0 or ys > y1)

def _annual_mean_window_on_target(nc_path: str, y0: int, y1: int) -> xr.DataArray:
    ds = xr.open_dataset(nc_path)
    try:
        var = _varname(ds)
        latn, lonn = _coords(ds)
        da = ds[var]
        if "time" not in da.dims:
            raise ValueError("variable has no time dimension")

        years = xr.DataArray(da["time"].dt.year, dims=("time",))
        sel = da.where((years >= y0) & (years <= y1), drop=True)
        if sel.sizes.get("time", 0) == 0:
            raise ValueError("no samples in requested window")

        da_y = sel.mean(dim="time", skipna=True)
        # alle Nicht-(lat/lon)-Dims wegreduzieren
        for d in list(da_y.dims):
            if d not in {latn, lonn}:
                da_y = da_y.mean(dim=d, skipna=True)

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

def _area_weights(lat: np.ndarray) -> np.ndarray:
    # cos(lat) Gewichtung (rad)
    return np.cos(np.deg2rad(lat)).astype("float32")

def _country_id(mask_ds: xr.Dataset, alpha2: str) -> int:
    inv = json.loads(mask_ds.attrs.get("iso2_lookup", "{}"))  # id -> iso2
    # invertieren: iso2 -> id
    fwd = {v.upper(): k for k, v in inv.items()}
    cid = fwd.get(alpha2.upper())
    if cid is None:
        raise ValueError(f"Country {alpha2} not found in mask")
    return int(cid)

def _country_weighted_mean(field: xr.DataArray, mask: xr.Dataset, alpha2: str) -> float:
    cid = _country_id(mask, alpha2)
    code = mask["code"].values  # (lat, lon)
    sel = (code == cid)
    if not np.any(sel):
        raise ValueError(f"No grid cells for country {alpha2} in mask")

    vals = field.values  # (lat, lon)
    wlat = _area_weights(mask["lat"].values)  # (lat,)
    # expand to (lat, lon)
    w = np.repeat(wlat[:, None], vals.shape[1], axis=1)
    w = np.where(sel, w, 0.0)
    v = np.where(sel, vals, np.nan)

    # gewichteten Mittelwert robust bilden
    sw = np.nansum(w)
    if sw <= 0:
        return float("nan")
    return float(np.nansum(v * w) / sw)

def compute_region_shock(
        *,
        client: Minio,
        bucket: str,
        alpha2: str,
        scenario: str,
        y0: int,
        y1: int,
        mode: Literal["baseline", "percentile"] = "baseline",
        q: float = 0.95,
        basis: str = "ensemble2015",
        agg: Literal["mean", "median"] = "median",
        stat: Literal["ratio", "diff"] = "ratio",
        mask_path: str = DEFAULT_MASK_PATH,
) -> Dict[str, Any]:
    """
    1) Lade/prüfe Ländermaske.
    2) Baue Proj.-Feld für [y0..y1] je Modell → regridded 1° → stack(member).
    3) Aggregiere über member (mean/median) zu 'proj_country'.
    4) Lade Threshold-Map (baseline / q).
    5) Aggregiere baseline/q über Land.
    6) shock = proj_country / base_country - 1  (oder diff).
    """
    # 1) Mask
    if not os.path.exists(mask_path):
        raise FileNotFoundError("Country mask not found. Run POST /api/pm25/country_mask/ensure first.")
    mds = xr.open_dataset(mask_path)

    try:
        # 2) Kandidaten
        index = _load_idx_json()
        files: List[Dict[str, Any]] = index.get("files", [])
        cand = [f for f in files if (f.get("scenario", "") == scenario and _covers_year_range(f, y0, y1))]
        if not cand:
            raise ValueError(f"No files for scenario={scenario} covering {y0}-{y1}")

        fields: List[xr.DataArray] = []
        models: Set[str] = set()
        errors: List[Dict[str, str]] = []

        for f in cand:
            key = f.get("key")
            try:
                local = _fetch_nc(client, bucket, key)
                da = _annual_mean_window_on_target(local, y0, y1)
                if not np.isfinite(da.values).any():
                    raise ValueError("field all-NaN after preprocessing")
                model = f.get("model") or "unknown"
                fields.append(da.expand_dims(member=[model]))
                models.add(model)
            except Exception as ex:
                errors.append({"key": key, "error": str(ex)})

        if not fields:
            raise ValueError(f"No usable fields for {scenario} {y0}-{y1}; first_errors={errors[:3]}")

        stack = xr.concat(fields, dim="member", join="outer", fill_value=np.nan)

        if agg == "mean":
            proj = stack.mean(dim="member", skipna=True)
        else:
            proj = stack.median(dim="member", skipna=True)  # default

        # 3) Landeswert projiziert
        proj_val = _country_weighted_mean(proj, mds, alpha2)

        # 4) Threshold-Map laden
        thr_path = os.path.join(CACHE_DIR, f"thrmap_{basis}_{scenario}_q{q:.2f}.nc")
        if not os.path.exists(thr_path):
            raise FileNotFoundError(
                f"Threshold map not found at {thr_path}. "
                f"Call GET /api/pm25/threshold_map?scenario={scenario}&quantile={q}&basis={basis} first."
            )

        tds = xr.open_dataset(thr_path)
        try:
            base_map = tds["baseline"]
            q_map = tds["q"]
            base_val = _country_weighted_mean(base_map, mds, alpha2)
            q_val = _country_weighted_mean(q_map, mds, alpha2)
        finally:
            tds.close()

        ref = base_val if mode == "baseline" else q_val

        # 5) Schock
        if stat == "diff":
            shock = float(proj_val - ref)
        else:
            # ratio
            shock = float(proj_val / ref) - 1.0 if (ref is not None and np.isfinite(ref) and ref != 0) else float("nan")

        return {
            "region": alpha2.upper(),
            "scenario": scenario,
            "window": [y0, y1],
            "mode": mode,
            "q": q,
            "basis": basis,
            "agg": agg,
            "stat": stat,
            "value": shock,
            "proj_country": proj_val,
            "ref_country": ref,
            "n_members": int(stack.sizes.get("member", 0)),
            "n_models": len(models),
        }
    finally:
        mds.close()
