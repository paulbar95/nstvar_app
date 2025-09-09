# service/shock.py
import os
import json
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Set, Optional, Literal

import numpy as np
import xarray as xr
from minio import Minio

from service.country_mask import DEFAULT_MASK_PATH
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

def _area_weights(lat: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(lat)).astype("float32")

def _country_id(mask_ds: xr.Dataset, alpha2: str) -> int:
    inv = json.loads(mask_ds.attrs.get("iso2_lookup", "{}"))  # id -> iso2
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
    w = np.repeat(wlat[:, None], vals.shape[1], axis=1)
    w = np.where(sel, w, 0.0)
    v = np.where(sel, vals, np.nan)

    sw = np.nansum(w)
    if sw <= 0:
        return float("nan")
    return float(np.nansum(v * w) / sw)

def _window_mean_on_target_from_paths(paths: List[str], y0: int, y1: int) -> xr.DataArray:
    """Öffnet mehrere Segmentdateien eines Members, konkatten nach Zeit, mittelt Fenster, regriddet auf 1°."""
    if len(paths) == 1:
        ds = xr.open_dataset(paths[0])
    else:
        # robust gegen segmentierte Zeitachsen
        ds = xr.open_mfdataset(paths, combine="by_coords", parallel=False)
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
    Korrekte Behandlung segmentierter CMIP-Dateien: je (model, run) werden ALLE Segmente geladen,
    zu einer Zeitreihe zusammengeführt und erst dann das Fenster [y0..y1] gemittelt.
    """
    # 1) Maske
    if not os.path.exists(mask_path):
        raise FileNotFoundError("Country mask not found. Run POST /api/pm25/country_mask/ensure first.")
    mds = xr.open_dataset(mask_path)

    try:
        # 2) Index → Kandidaten gruppieren
        index = _load_idx_json()
        files: List[Dict[str, Any]] = index.get("files", [])
        cand = [f for f in files if (f.get("scenario", "") == scenario and _covers_year_range(f, y0, y1))]
        if not cand:
            raise ValueError(f"No files for scenario={scenario} covering {y0}-{y1}")

        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for f in cand:
            model = f.get("model") or "unknown"
            run = f.get("run") or "r?"
            groups[(model, run)].append(f)

        fields: List[xr.DataArray] = []
        members_labels: List[str] = []
        models: Set[str] = set()
        errors: List[Dict[str, str]] = []

        for (model, run), segs in groups.items():
            try:
                # alle Segmente des Members, die in den Zeitraum fallen
                seg_keys = [s.get("key") for s in segs if s.get("key")]
                if not seg_keys:
                    continue
                local_paths = [_fetch_nc(client, bucket, k) for k in seg_keys]
                da = _window_mean_on_target_from_paths(local_paths, y0, y1)
                if not np.isfinite(da.values).any():
                    raise ValueError("field all-NaN after preprocessing")
                member_id = f"{model}:{run}"
                fields.append(da.expand_dims(member=[member_id]))
                members_labels.append(member_id)
                models.add(model)
            except Exception as ex:
                errors.append({"member": f"{model}:{run}", "error": str(ex)})

        if not fields:
            raise ValueError(f"No usable members for {scenario} {y0}-{y1}; first_errors={errors[:3]}")

        stack = xr.concat(fields, dim="member", join="outer", fill_value=np.nan)
        proj = stack.mean(dim="member", skipna=True) if agg == "mean" else stack.median(dim="member", skipna=True)
        proj_val = _country_weighted_mean(proj, mds, alpha2)

        # 3) Threshold-Map laden
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

        if stat == "diff":
            shock = float(proj_val - ref)
        else:
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
            "n_members": int(stack.sizes.get("member", 0)),        # ~ Anzahl (model,run)
            "members": members_labels[:50],
            "n_models": len(models),
            "errors": errors[:5],
        }
    finally:
        mds.close()
