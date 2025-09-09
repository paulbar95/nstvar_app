import os, json, hashlib
from typing import Dict, Any, List, Tuple

import numpy as np
import xarray as xr
from minio import Minio
from minio.error import S3Error

INDEX_PATH = "/app/cache/pm25_index.json"
CACHE_DIR = "/app/cache"
NC_CACHE_DIR = "/app/cache/nc"

def _ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(NC_CACHE_DIR, exist_ok=True)

def _minio() -> Minio:
    return Minio(
        os.environ.get("MINIO_URL", "minio:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )

def _load_index() -> Dict[str, Any]:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(INDEX_PATH)
    with open(INDEX_PATH, "r") as f:
        return json.load(f)

def _parse_filelist(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    return index.get("files") if isinstance(index, dict) and "files" in index else index

def _covers_year(f: Dict[str, Any], year: int) -> bool:
    s, e = f.get("start"), f.get("end")
    return bool(s and e and s <= f"{year}01" and e >= f"{year}12")

def _is_scenario(f: Dict[str, Any], scenario: str) -> bool:
    return (f.get("scenario") or "").lower() == scenario.lower()

def _is_historical(f: Dict[str, Any]) -> bool:
    return (f.get("scenario") or "").lower() == "historical"

def _nc_cache_key(bucket: str, key: str) -> str:
    h = hashlib.sha256(f"{bucket}/{key}".encode("utf-8")).hexdigest()[:16]
    base = key.split("/")[-1]
    return os.path.join(NC_CACHE_DIR, f"{h}_{base}")

def _download_nc(client: Minio, bucket: str, key: str) -> str:
    _ensure_dirs()
    path = _nc_cache_key(bucket, key)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        client.fget_object(bucket, key, path)
    return path

def _varname(ds: xr.Dataset) -> str:
    for c in ("mmrpm2p5", "pm2p5", "pm25"):
        if c in ds.data_vars:
            return c
    # Fallback: erstes Datenfeld
    if ds.data_vars:
        return list(ds.data_vars)[0]
    raise ValueError("No data variables in dataset")

def _coords(ds: xr.Dataset) -> Tuple[str, str]:
    for la in ("lat", "latitude"):
        if la in ds.coords:
            for lo in ("lon", "longitude"):
                if lo in ds.coords:
                    return la, lo
    raise ValueError("No lat/lon coordinates in dataset")

def _target_grid() -> Tuple[np.ndarray, np.ndarray]:
    # 1° Grid, global
    lats = np.arange(-89.5, 90.5, 1.0)
    lons = np.arange(0.5, 360.5, 1.0)  # [0,360)
    return lats, lons

def _annual_mean_2015_on_target(nc_path: str) -> xr.DataArray:
    ds = xr.open_dataset(nc_path)
    try:
        var = _varname(ds)
        latn, lonn = _coords(ds)
        da = ds[var]
        if "time" not in da.dims:
            raise ValueError("variable has no time dimension")

        # ROBUSTER: über .dt.year filtern (funktioniert auch mit cftime)
        years = xr.DataArray(da["time"].dt.year, dims=("time",))
        da_2015 = da.where(years == 2015, drop=True)
        if da_2015.sizes.get("time", 0) == 0:
            raise ValueError("no samples with year==2015")

        da_y = da_2015.mean(dim="time", skipna=True)

        # Regridding via interp auf Zielgrid (mit 0..360 Behandlung)
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

def _annual_means_historical_on_target(nc_path: str, y0=1850, y1=1900) -> xr.DataArray:
    ds = xr.open_dataset(nc_path)
    var = _varname(ds)
    latn, lonn = _coords(ds)
    da = ds[var]

    # Jahresmittel über [y0..y1]
    da_sel = da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    if da_sel.sizes.get("time", 0) == 0:
        ds.close()
        raise ValueError(f"No data in historical slice {y0}-{y1} for {nc_path}")

    # Monatsmittel → Jahresmittel
    if "time" in da_sel.dims:
        da_y = da_sel.groupby("time.year").mean(dim="time", skipna=True)
    else:
        ds.close()
        raise ValueError("No time dimension")

    # Interp auf Zielgrid
    tg_lat, tg_lon = _target_grid()
    lon = ds[lonn]
    lon_vals = lon.values
    if lon_vals.min() < 0:
        lon_vals = (lon_vals + 360) % 360
        da_y = da_y.assign_coords({lonn: lon_vals}).sortby(lonn)

    da_t = da_y.interp(
        {latn: xr.DataArray(tg_lat, dims=("lat",)), lonn: xr.DataArray(tg_lon, dims=("lon",))},
        method="nearest"
    )
    da_t = da_t.rename({latn: "lat", lonn: "lon"})
    ds.close()
    return da_t  # dims: year, lat, lon

def _save_nc(path: str, ds: xr.Dataset):
    _ensure_dirs()
    ds.to_netcdf(path)

def build_threshold_map(scenario: str, q: float, basis: str, force: bool=False) -> Dict[str, Any]:
    """
    Erzeuge zellweise Threshold-Karte + Baseline:
      - basis='ensemble2015': q über 2015-Modelle je Zelle
      - basis='historical'   : q über Jahresmittel 1850-1900 (Zeit×Modelle) je Zelle
    Speichert Datei unter /app/cache/thrmap_{basis}_{scenario}_q{q:.2f}.nc
    """
    _ensure_dirs()
    if not (0.0 <= q <= 1.0):
        raise ValueError("quantile must be in [0,1]")
    out_nc = os.path.join(CACHE_DIR, f"thrmap_{basis}_{scenario}_q{q:.2f}.nc")
    if os.path.exists(out_nc) and not force:
        return {"path": out_nc, "cached": True}

    index = _load_index()
    files = _parse_filelist(index)
    bucket = os.environ.get("PM25_BUCKET", "pm25data")
    client = _minio()

    tg_lat, tg_lon = _target_grid()

    # Sammeln der Felder auf Zielgrid
    fields: List[xr.DataArray] = []
    used = 0
    models = set()
    errors = []

    if basis == "ensemble2015":
        candidate_files = [f for f in files if _is_scenario(f, scenario) and _covers_year(f, 2015)]
        for f in candidate_files:
            key = f.get("key") or f.get("name") or f.get("Key")
            if not key:
                continue
            try:
                nc = _download_nc(client, bucket, key)
                da = _annual_mean_2015_on_target(nc)
                fields.append(da)
                used += 1
                if f.get("model"): models.add(f["model"])
            except Exception as ex:
                errors.append({"key": key, "error": str(ex)})
                continue

        if not fields:
            # Gib die ersten Fehler mit zurück – viel hilfreicher beim Debuggen
            raise ValueError(f"No 2015 fields for scenario {scenario}; first_errors={errors[:3]}")

    elif basis == "historical":
        candidate_files = [f for f in files if _is_historical(f)]
        for f in candidate_files:
            key = f.get("key") or f.get("name") or f.get("Key")
            if not key:
                continue
            try:
                nc = _download_nc(client, bucket, key)
                da_y = _annual_means_historical_on_target(nc, 1850, 1900)  # dims: year, lat, lon
                fields.append(da_y)
                used += 1
                if f.get("model"): models.add(f["model"])
            except Exception:
                continue

        if not fields:
            raise ValueError("No historical (1850-1900) annual means available")

        stack = xr.concat(fields, dim="source")  # dims: source, year, lat, lon
        # baseline: als Referenz B_c = Mittel über 2015 brauchen wir trotzdem für Pineau-Schock;
        # hier setzen wir baseline als Mittel über historische Jahre (optional),
        # der eigentliche Schock nutzt später die 2015-Baseline.
        baseline = stack.mean(dim=("source","year"), skipna=True)
        # Quantil über alle Jahreswerte + Quellen
        qmap = stack.stack(sample=("source","year")).quantile(q, dim="sample", skipna=True)

    else:
        raise ValueError("basis must be 'ensemble2015' or 'historical'")

    ds_out = xr.Dataset(
        data_vars=dict(
            baseline=baseline,  # lat, lon
            q=qmap,             # lat, lon
            count=xr.DataArray(np.full((len(tg_lat), len(tg_lon)), used, dtype=np.int32), dims=("lat","lon"))
        ),
        coords=dict(
            lat=("lat", tg_lat),
            lon=("lon", tg_lon)
        ),
        attrs=dict(
            scenario=scenario,
            basis=basis,
            quantile=float(q),
            models=",".join(sorted(models)),
            n_inputs=int(used)
        )
    )
    _save_nc(out_nc, ds_out)

    # kleine Preview-Stats zurückgeben
    qvals = ds_out["q"].values
    stats = dict(
        min=float(np.nanmin(qvals)),
        p50=float(np.nanmedian(qvals)),
        max=float(np.nanmax(qvals))
    )
    return {
        "path": out_nc,
        "cached": False,
        "n_inputs": used,
        "n_models": len(models),
        "stats_q": stats,
        "basis": basis,
        "scenario": scenario
    }
