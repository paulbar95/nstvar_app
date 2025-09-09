# service/region_mask.py
import os
from typing import Tuple

import numpy as np
import xarray as xr

from service.threshold_map import _target_grid, _coords

STATIC_DIR = "/app/static"
MASK_PATH = os.path.join(STATIC_DIR, "country_mask_1deg.nc")  # erwartet Variable "iso2"


def _area_weights_lat(lat: np.ndarray) -> np.ndarray:
    # einfache sphärische Fläche ~ cos(lat)
    return np.cos(np.deg2rad(lat))


def load_country_mask() -> xr.Dataset:
    """
    Erwartet /app/static/country_mask_1deg.nc mit:
      dims: lat(180), lon(360)
      coords: lat [-89.5..+89.5], lon [0.5..359.5]
      data_var: iso2 (string oder bytes) — ISO-3166-1 alpha-2
    """
    if not os.path.exists(MASK_PATH):
        raise FileNotFoundError(
            f"Country mask not found: {MASK_PATH}. "
            "Provide a 1°-grid mask with variable 'iso2'."
        )
    ds = xr.open_dataset(MASK_PATH)
    # auf Standardnamen bringen (falls anders benannt)
    try:
        latn, lonn = _coords(ds)
        if latn != "lat" or lonn != "lon":
            ds = ds.rename({latn: "lat", lonn: "lon"})
        return ds
    except Exception:
        ds.close()
        raise


def region_mask(alpha2: str) -> xr.DataArray:
    """
    Gibt boolesche Maske (lat,lon) für das 1°-Grid zurück (True = in Region).
    """
    ds = load_country_mask()
    try:
        iso = ds["iso2"]
        vals = iso.values
        # robust gegen bytes/objects
        if vals.dtype.kind in ("S", "O"):
            vals = vals.astype(str)
        mask = (vals == alpha2.upper())
        da = xr.DataArray(
            mask,
            dims=("lat", "lon"),
            coords={"lat": ds["lat"].values, "lon": ds["lon"].values},
            name="mask",
        )
        # sicherstellen, dass die Maske exakt auf unser Zielgrid passt
        tg_lat, tg_lon = _target_grid()
        if (not np.array_equal(da["lat"].values, tg_lat)) or (not np.array_equal(da["lon"].values, tg_lon)):
            da = da.interp(
                {"lat": xr.DataArray(tg_lat, dims=("lat",)),
                 "lon": xr.DataArray(tg_lon, dims=("lon",))},
                method="nearest"
            )
        return da
    finally:
        ds.close()


def area_weighted_mean(field: xr.DataArray, mask: xr.DataArray) -> Tuple[float, int, int]:
    """
    Flächengewichtetes Mittel über maskierte Zellen.
    field: xr.DataArray(lat,lon)
    mask:  xr.DataArray(lat,lon) bool
    Returns: (value, n_valid, n_mask)
    """
    # align
    field, mask = xr.align(field, mask, join="inner")

    lat = field["lat"].values
    w_lat = _area_weights_lat(lat)  # shape (lat,)
    w2d = xr.DataArray(
        np.broadcast_to(w_lat[:, None], field.shape),
        dims=("lat", "lon"),
        coords={"lat": field["lat"], "lon": field["lon"]},
    )

    valid = np.isfinite(field.values)
    used = (mask.values.astype(bool)) & valid
    n_mask = int(mask.values.sum())
    n_used = int(used.sum())

    if n_used == 0:
        return (float("nan"), n_used, n_mask)

    num = (field.where(used) * w2d.where(used)).sum().item()
    den = (w2d.where(used)).sum().item()
    return (float(num / den), n_used, n_mask)
