# service/country_mask.py
import json
import os
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
import urllib.request
from shapely.geometry import shape, Point, Polygon, MultiPolygon

CACHE_DIR = "/app/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DEFAULT_MASK_PATH = os.path.join(CACHE_DIR, "country_mask_1deg.nc")
DEFAULT_GEOJSON_PATH = os.path.join(CACHE_DIR, "ne_countries_lowres.geojson")
# Natural Earth v5.1.2 – Admin 0 - Countries (Low resolution, 110m)
DEFAULT_GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
)

# ---------- Zielraster (1°) ----------
def target_grid() -> Tuple[np.ndarray, np.ndarray]:
    """
    1°-Raster, wie in threshold_map.py (kompatibel).
    lat: -89.5..+89.5 (180)
    lon: 0.5..359.5 (360)
    """
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
    # Natural Earth Feld-Reihenfolge: bevorzugt ISO_A2_EH, sonst ISO_A2
    props = feat.get("properties", {})
    iso2 = props.get("ISO_A2_EH") or props.get("ISO_A2")
    if not iso2 or iso2 == "-99":
        return None
    iso2 = iso2.strip()
    if len(iso2) != 2:
        return None
    # Bsp. 'FR' -> 'FR'
    return iso2.upper()


def _geometry_from_feature(feat: Dict) -> Optional[MultiPolygon | Polygon]:
    geom = feat.get("geometry")
    if not geom:
        return None
    g = shape(geom)
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    return None


def _build_mask_from_geojson(geojson_path: str) -> xr.Dataset:
    """
    Rastert Länder-Polygone auf 1°-Raster.
    Speichert den Länder-Code als "code" (int16) und ein Lookup Mapping (Attr).
    """
    lat, lon = target_grid()
    nlat, nlon = lat.size, lon.size
    # mask int16; 0 = no country; 1..N = ISO2 index
    code = np.zeros((nlat, nlon), dtype=np.int16)

    data = _load_geojson(geojson_path)
    features = data.get("features", [])

    # ISO2 -> int index
    idx_by_iso2: Dict[str, int] = {}
    next_idx = 1  # 0 = no-data

    # Vorbereite Punkt-Liste je Spalte (lon) – wir erzeugen Points on-the-fly
    # Bounding Box Filter spart viel Zeit.
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
        # Koordinaten auf 0..360 longitudes bringen, falls GeoJSON bei -180..180 liegt
        if minx < 0 or maxx <= 180:
            # wir nehmen an, geometries sind -180..180; bring sie in 0..360
            def _wrap_x(x):
                v = (x + 360.0) % 360.0
                return v

            # bounding box in 0..360
            minx = _wrap_x(minx)
            maxx = _wrap_x(maxx)
            # Achtung: bei Bounding-Box, die den Meridian schneidet, ist das nicht perfekt.
            # Für Low-Res reicht es – in Zweifelsfällen prüft der "contains" am Ende sauber.

        # Indexbereiche grob einschränken
        lat_mask = (lat_vals >= (miny - 1.0)) & (lat_vals <= (maxy + 1.0))
        if minx <= maxx:
            lon_mask = (lon_vals >= (minx - 1.0)) & (lon_vals <= (maxx + 1.0))
            lon_indices = np.where(lon_mask)[0]
        else:
            # Wrap-around (selten)
            lon_mask = (lon_vals >= (minx - 1.0)) | (lon_vals <= (maxx + 1.0))
            lon_indices = np.where(lon_mask)[0]

        lat_indices = np.where(lat_mask)[0]

        # Punkt-in-Polygon
        for ii in lat_indices:
            y = float(lat_vals[ii])
            for jj in lon_indices:
                # schon zugewiesen? wir lassen "erstes" Land gewinnen; reicht für 1° low-res
                if code[ii, jj] != 0:
                    continue
                x = float(lon_vals[jj])
                p = Point(x, y)
                if geom.contains(p):
                    code[ii, jj] = cid

    # Dataset schreiben
    # Mapping int->ISO2 als JSON-Attribut ablegen
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
        src: Optional[str] = None,        # URL ODER lokaler Pfad (GeoJSON)
        overwrite: bool = False
) -> Dict[str, object]:
    """
    Stellt sicher, dass eine 1°-Länder-Maske existiert.
    - Wenn `mask_path` existiert und `overwrite=False` -> nur Metadaten zurück.
    - Sonst: `src` (URL oder Pfad) verwenden. Fehlt `src`, laden wir Natural Earth Low-Res.
    """
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

    if os.path.exists(mask_path) and not overwrite:
        ds = xr.open_dataset(mask_path)
        try:
            iso_attr = ds.attrs.get("iso2_lookup", "{}")
            inv = json.loads(iso_attr)
            return {
                "path": mask_path,
                "created": False,
                "n_countries": len(inv),
                "shape": [int(ds.dims["lat"]), int(ds.dims["lon"])],
            }
        finally:
            ds.close()

    # Quelle besorgen
    if src is None:
        # Standard: Natural Earth
        geojson_path = _download_if_needed(DEFAULT_GEOJSON_URL, DEFAULT_GEOJSON_PATH)
    else:
        if src.startswith("http://") or src.startswith("https://"):
            geojson_path = _download_if_needed(src, DEFAULT_GEOJSON_PATH)
        else:
            geojson_path = src  # lokaler Pfad

    ds_mask = _build_mask_from_geojson(geojson_path)
    # überschreiben ok
    if os.path.exists(mask_path):
        os.remove(mask_path)
    ds_mask.to_netcdf(mask_path)

    inv = json.loads(ds_mask.attrs.get("iso2_lookup", "{}"))
    return {
        "path": mask_path,
        "created": True,
        "n_countries": len(inv),
        "shape": [int(ds_mask.dims["lat"]), int(ds_mask.dims["lon"])],
        "source": geojson_path,
    }
