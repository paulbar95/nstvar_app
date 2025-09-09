# service/country_mask.py
import os
import json
import urllib.request
from typing import Dict, Any, List, Tuple

import numpy as np
import xarray as xr
from shapely.geometry import shape, Point
from shapely.geometry.base import BaseGeometry

CACHE_DIR = "/app/cache"
STATIC_DIR = "/app/static"

DEFAULT_GEOJSON_PATH = os.getenv("COUNTRY_GEOJSON", f"{STATIC_DIR}/ne_110m_admin_0_countries.geojson")
DEFAULT_GEOJSON_URL  = os.getenv(
    "COUNTRY_GEOJSON_URL",
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
)
DEFAULT_MASK_PATH = os.path.join(CACHE_DIR, "country_mask_1deg.nc")

# --- interne Version (historisch) ---
def _target_grid() -> Tuple[np.ndarray, np.ndarray]:
    # 1°-Grid (lat: -89.5..89.5; lon: 0.5..359.5)
    lat = np.linspace(-89.5, 89.5, 180)
    lon = np.linspace(0.5, 359.5, 360)
    return lat, lon

# --- öffentlicher Alias, damit validate.py importieren kann ---
def target_grid() -> Tuple[np.ndarray, np.ndarray]:
    return _target_grid()

def has_mask(path: str = DEFAULT_MASK_PATH) -> bool:
    return os.path.exists(path)

def ensure_country_mask(path: str = DEFAULT_MASK_PATH) -> str:
    """
    Stellt sicher, dass eine Länder-Maske existiert.
    Aktuell: Fallback-Dummy (leere Strings) – verhindert Crashs, bis die echte
    Geometrie-Variante aktiv ist. Ersetze das später durch die GeoJSON/regionmask-Logik.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

    if os.path.exists(path):
        return path

    lat, lon = target_grid()
    data = xr.DataArray(
        np.full((lat.size, lon.size), "", dtype=object),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="iso_a2",
    )
    xr.Dataset({"iso_a2": data}).to_netcdf(path)
    return path

def _ensure_dirs():
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def _download_geojson(url: str, dest: str) -> None:
    _ensure_dirs()
    urllib.request.urlretrieve(url, dest)

def _load_geojson(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    # FeatureCollection erwartet
    feats = gj["features"] if "features" in gj else []
    return feats

def _iso2_of_feature(feat: Dict[str, Any]) -> str | None:
    props = feat.get("properties", {})
    # Natural Earth Felder (häufig):
    for k in ("ISO_A2", "ADM0_A3", "iso_a2", "abbrev"):
        v = props.get(k)
        if isinstance(v, str) and len(v) >= 2:
            code = v[:2].upper()
            if code.isalpha():
                return code
    return None

def _polygons_from_feature(feat: Dict[str, Any]) -> BaseGeometry:
    geom = feat.get("geometry")
    if not geom:
        raise ValueError("feature without geometry")
    return shape(geom)  # shapely Geometry (Polygon/MultiPolygon)

def build_country_mask(geojson_path: str, mask_path: str) -> Dict[str, Any]:
    """Rastert Länder-Polygone auf 1°-Grid. Ergebnis:
       Dataset mit dims: country, lat, lon
       vars: mask (bool), coord 'country' (ISO2 strings)
    """
    _ensure_dirs()
    feats = _load_geojson(geojson_path)

    # Sammle Polygone pro ISO2
    by_iso: dict[str, List[BaseGeometry]] = {}
    for ft in feats:
        code = _iso2_of_feature(ft)
        if not code:
            continue
        geom = _polygons_from_feature(ft)
        by_iso.setdefault(code, []).append(geom)

    lat, lon = _target_grid()
    nlat, nlon = len(lat), len(lon)
    countries = sorted(by_iso.keys())
    nc = len(countries)

    # Output-Array
    mask = np.zeros((nc, nlat, nlon), dtype=np.bool_)

    # Prüfpunkte: Zellzentren (lon 0..360 → für Point-in-Polygon in [-180..180] normalisieren)
    lon_deg = lon.copy()
    lon_west_east = np.where(lon_deg > 180.0, lon_deg - 360.0, lon_deg)  # -180..180

    # Rasterung (einfach: Punkt-in-Polygon am Zellzentrum)
    for ci, iso in enumerate(countries):
        # vereinige MultiPolygone dieses Landes
        geoms = by_iso[iso]
        # shapely kann Sammlung als unary_union vereinigen (optional)
        geom_union = geoms[0]
        for g in geoms[1:]:
            try:
                geom_union = geom_union.union(g)
            except Exception:
                # robust: wenn union scheitert, nimm beide (contain-check funktioniert auch so)
                pass

        for yi, la in enumerate(lat):
            for xi, lo in enumerate(lon_west_east):
                p = Point(float(lo), float(la))
                if geom_union.contains(p):
                    mask[ci, yi, xi] = True

    # als Dataset speichern
    ds = xr.Dataset(
        data_vars={
            "mask": (("country", "lat", "lon"), mask),
        },
        coords={
            "country": np.array(countries, dtype=object),  # string-koordinate
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "grid": "1deg",
            "lon_convention": "0..360 (cell centers), selection uses same",
            "geojson_source": os.path.abspath(geojson_path),
        },
    )
    if os.path.exists(mask_path):
        os.remove(mask_path)
    ds.to_netcdf(mask_path)
    ds.close()

    return {
        "mask_path": mask_path,
        "n_countries": len(countries),
        "countries": countries[:10],  # Vorschau
    }

def mask_info(mask_path: str | None = None) -> Dict[str, Any]:
    p = mask_path or DEFAULT_MASK_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    ds = xr.open_dataset(p)
    try:
        return {
            "mask_path": p,
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "countries": [str(c) for c in ds["country"].values[:20]],
        }
    finally:
        ds.close()
