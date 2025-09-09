# service/country_mask.py
import json
import os
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from shapely.geometry import shape, Point, Polygon, MultiPolygon

CACHE_DIR = "/app/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_MASK_PATH = os.path.join(CACHE_DIR, "country_mask_1deg.nc")
DEFAULT_GEOJSON_PATH = os.path.join(CACHE_DIR, "ne_countries_lowres.geojson")
DEFAULT_GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
)

def target_grid() -> Tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(-89.5, 89.5, 180)
    lon = np.linspace(0.5, 359.5, 360)
    return lat, lon

def _download_if_needed(src_url: str, to_path: str) -> str:
    os.makedirs(os.path.dirname(to_path), exist_ok=True)
    if not os.path.exists(to_path):
        urllib.request.urlretrieve(src_url, to_path)
    return to_path

def _load_geojson(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _iso2_from_feature(feat: Dict) -> Optional[str]:
    props = feat.get("properties", {})
    iso2 = props.get("ISO_A2_EH") or props.get("ISO_A2")
    if not iso2 or iso2 == "-99":
        return None
    iso2 = iso2.strip().upper()
    return iso2 if len(iso2) == 2 else None

def _geometry_from_feature(feat: Dict) -> Optional[MultiPolygon | Polygon]:
    geom = feat.get("geometry")
    if not geom:
        return None
    g = shape(geom)
    return g if isinstance(g, (Polygon, MultiPolygon)) else None

def _build_mask_from_geojson(geojson_path: str) -> xr.Dataset:
    lat, lon = target_grid()
    nlat, nlon = lat.size, lon.size
    code = np.zeros((nlat, nlon), dtype=np.int16)

    data = _load_geojson(geojson_path)
    features = data.get("features", [])

    idx_by_iso2: Dict[str, int] = {}
    next_idx = 1

    lon_vals = lon.copy()
    lat_vals = lat.copy()

    for feat in features:
        iso2 = _iso2_from_feature(feat)
        if not iso2:
            continue
        geom = _geometry_from_feature(feat)
        if geom is None:
            continue

        if iso2 not in idx_by_iso2:
            idx_by_iso2[iso2] = next_idx
            next_idx += 1
        cid = idx_by_iso2[iso2]

        minx, miny, maxx, maxy = geom.bounds
        if minx < 0 or maxx <= 180:
            def _wrap_x(x): return (x + 360.0) % 360.0
            minx = _wrap_x(minx); maxx = _wrap_x(maxx)

        lat_mask = (lat_vals >= (miny - 1.0)) & (lat_vals <= (maxy + 1.0))
        if minx <= maxx:
            lon_mask = (lon_vals >= (minx - 1.0)) & (lon_vals <= (maxx + 1.0))
            lon_indices = np.where(lon_mask)[0]
        else:
            lon_mask = (lon_vals >= (minx - 1.0)) | (lon_vals <= (maxx + 1.0))
            lon_indices = np.where(lon_mask)[0]
        lat_indices = np.where(lat_mask)[0]

        for ii in lat_indices:
            y = float(lat_vals[ii])
            for jj in lon_indices:
                if code[ii, jj] != 0:
                    continue
                x = float(lon_vals[jj])
                if geom.contains(Point(x, y)):
                    code[ii, jj] = cid

    inv = {int(v): k for k, v in idx_by_iso2.items()}
    ds = xr.Dataset(
        {"code": (("lat", "lon"), code.astype(np.int16))},
        coords={"lat": lat_vals, "lon": lon_vals},
        attrs={"iso2_lookup": json.dumps(inv)},
    )
    return ds

def ensure_country_mask(
        mask_path: str = DEFAULT_MASK_PATH,
        *,
        src: Optional[str] = None,
        overwrite: bool = False
) -> Dict[str, object]:
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

    if os.path.exists(mask_path) and not overwrite:
        ds = xr.open_dataset(mask_path)
        try:
            inv = json.loads(ds.attrs.get("iso2_lookup", "{}"))
            return {
                "path": mask_path,
                "created": False,
                "n_countries": len(inv),
                "shape": [int(ds.sizes["lat"]), int(ds.sizes["lon"])],
            }
        finally:
            ds.close()

    if src is None:
        geojson_path = _download_if_needed(DEFAULT_GEOJSON_URL, DEFAULT_GEOJSON_PATH)
    else:
        if src.startswith("http://") or src.startswith("https://"):
            geojson_path = _download_if_needed(src, DEFAULT_GEOJSON_PATH)
        else:
            geojson_path = src

    ds_mask = _build_mask_from_geojson(geojson_path)
    if os.path.exists(mask_path):
        os.remove(mask_path)
    ds_mask.to_netcdf(mask_path)

    inv = json.loads(ds_mask.attrs.get("iso2_lookup", "{}"))
    return {
        "path": mask_path,
        "created": True,
        "n_countries": len(inv),
        "shape": [int(ds_mask.sizes["lat"]), int(ds_mask.sizes["lon"])],
        "source": geojson_path,
    }

def mask_info(mask_path: str = DEFAULT_MASK_PATH) -> Dict[str, object]:
    if not os.path.exists(mask_path):
        raise FileNotFoundError("Country mask not found")
    ds = xr.open_dataset(mask_path)
    try:
        inv = json.loads(ds.attrs.get("iso2_lookup", "{}"))
        return {
            "path": mask_path,
            "n_countries": len(inv),
            "shape": [int(ds.sizes["lat"]), int(ds.sizes["lon"])],
            "sample": dict(list(inv.items())[:5]),
        }
    finally:
        ds.close()
