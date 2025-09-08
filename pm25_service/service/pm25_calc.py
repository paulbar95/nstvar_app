from __future__ import annotations
import numpy as np
import xarray as xr
import regionmask, pycountry
from typing import Iterable, Dict, List
from .s3io import open_dataset_s3
from .normalize import pick_pm_var, normalize_da

def _area_weights(lat: xr.DataArray) -> xr.DataArray:
    # einfache Kosinus-Breite
    w = np.cos(np.deg2rad(lat))
    w = xr.DataArray(w, dims=("lat",))
    return w / w.mean()

def _country_mask(lat: xr.DataArray, lon: xr.DataArray, alpha2: str):
    reg = regionmask.defined_regions.natural_earth_v5_1_2.countries_110
    country = pycountry.countries.get(alpha_2=alpha2.upper())
    if not country or country.name not in reg.names:
        raise ValueError(f"Unknown/unsupported region '{alpha2}'.")
    idx = reg.names.index(country.name)
    mask = reg.mask(lon=lon.values, lat=lat.values)  # (lat,lon)
    if np.isnan(mask).all():
        raise ValueError("Region mask failed: no grid cells in region.")
    return mask == reg.numbers[idx]

def _mean_region(da: xr.DataArray, region_bool) -> xr.DataArray:
    # Gewichtet nach Breite, dann Regionmittel
    wlat = _area_weights(da["lat"])
    sel = da.where(region_bool)
    return (sel * wlat).sum(("lat","lon"), skipna=True) / (wlat * (~np.isnan(sel.isel(lon=0)))).sum("lat", skipna=True)

def compute_baseline_2015_per_model(index: Iterable[Dict]) -> Dict[str, xr.DataArray]:
    """
    Nimmt das erste File je Modell (egal welches SSP – 2015 ist identisch im ersten Jahr des SSP-Runs)
    und bildet das 2015-Jahresmittel als Baseline.
    Returns: dict model -> (lat,lon) DataArray
    """
    by_model: Dict[str, str] = {}
    for it in index:
        m = it["model"]
        # wähle ein SSP-File mit Start 201501 (so wie in deinem Index)
        if it.get("start") == "201501" and m not in by_model:
            by_model[m] = it["key"]

    out = {}
    for model, key in by_model.items():
        ds = open_dataset_s3(key)
        var = pick_pm_var(ds)
        da = normalize_da(ds[var])
        # 2015 Mittel – bei deinen Dateien beginnt time in 2015-01
        # Falls echte Zeitkoordinate fehlt: nimm die ersten 12 Schritte
        if "time" in da.dims and "time" in da.coords and np.issubdtype(da["time"].dtype, np.datetime64):
            base = da.sel(time=slice("2015-01-01", "2015-12-31")).mean("time", skipna=True)
        else:
            base = da.isel(time=slice(0,12)).mean("time", skipna=True)
        out[model] = base  # (lat,lon)
    if not out:
        raise RuntimeError("No baseline could be computed (no 2015 starts found).")
    return out

def compute_region_shock(index: Iterable[Dict], alpha2: str, scenario: str, y0: int, y1: int) -> Dict:
    """
    Schock = (Szenario-Mittel(y0..y1) / Baseline2015) - 1
    Liefert Ensemble-Mittel und Einzelmodelle.
    """
    # 1) Baselines je Modell
    baselines = compute_baseline_2015_per_model(index)

    # 2) Szenario-Dateien pro Modell einsammeln
    by_model_scen: Dict[str, str] = {}
    scen = scenario.lower()
    for it in index:
        if it["scenario"].lower() == scen and it["model"] in baselines:
            by_model_scen[it["model"]] = it["key"]

    if not by_model_scen:
        raise ValueError(f"No files for scenario '{scenario}' that match baseline models.")

    model_vals: List[float] = []

    for model, key in by_model_scen.items():
        ds = open_dataset_s3(key)
        var = pick_pm_var(ds)
        da = normalize_da(ds[var])

        # Zeitraum-Mittel
        if "time" in da.dims and "time" in da.coords and np.issubdtype(da["time"].dtype, np.datetime64):
            fut = da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31")).mean("time", skipna=True)
        else:
            # Fallback: Index-Annahme: start 2015-01 -> Jahresindex = (year-2015)*12
            t0 = max(0, (y0 - 2015) * 12)
            t1 = (y1 - 2015 + 1) * 12
            fut = da.isel(time=slice(t0, t1)).mean("time", skipna=True)

        base = baselines[model]  # (lat,lon), bereits normalisiert

        # Schock-Karte
        shock = (fut / base) - 1.0

        # Regionmittel
        mask_bool = _country_mask(shock["lat"], shock["lon"], alpha2)
        val = _mean_region(shock, mask_bool).item()
        model_vals.append(float(val))

    ens_mean = float(np.nanmean(model_vals)) if model_vals else float("nan")
    return {
        "region": alpha2.upper(),
        "scenario": scenario,
        "years": [y0, y1],
        "ensemble_mean": ens_mean,
        "models": [{"model": m, "value": v} for m, v in zip(by_model_scen.keys(), model_vals)]
    }
