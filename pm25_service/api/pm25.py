from fastapi import APIRouter, Query, HTTPException
import os, json  # <— fehlte
from service.region_value import get_region_value
from service.threshold import get_threshold
from service.shock import compute_region_shock
from service.validate import check_mask_vs_target, check_alignment_with_sample
from typing import Optional
from minio import Minio

from typing import Optional
from fastapi import Body
from service.country_mask import ensure_country_mask, DEFAULT_MASK_PATH

# falls du region_shock sofort nutzen willst:
from service.indexer import index_pm25_data         # <— damit kein NameError
from service.threshold_map import build_threshold_map

from service.threshold_map import build_threshold_map
from service.window_field import window_field
from service.shock import compute_region_shock

router = APIRouter(tags=["PM2.5"])

INDEX_PATH = "/app/cache/pm25_index.json"

def _minio() -> Minio:
    return Minio(
        os.environ.get("MINIO_URL", "minio:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )

DEFAULT_BUCKET = os.getenv("PM25_BUCKET", "pm25data")

@router.post("/country_mask/ensure")
def ensure_mask(
        src: str | None = Query(
            None,
            description="Optional: URL oder lokaler Pfad zu einem Countries-GeoJSON. "
                        "Wenn leer, wird Natural Earth Low-Res automatisch verwendet."
        ),
        overwrite: bool = Query(False, description="Vorhandene Maske überschreiben")
):
    try:
        res = ensure_country_mask(mask_path=DEFAULT_MASK_PATH, src=src, overwrite=overwrite)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/country_mask/info")
def country_mask_info():
    try:
        return mask_info_fn()
    except FileNotFoundError:
        raise HTTPException(404, "Mask not found. Run POST /api/pm25/country_mask/ensure first.")

@router.get("/region_shock")
def region_shock(
        region: str = Query(..., min_length=2, max_length=2, description="ISO-3166-1 alpha-2"),
        scenario: str = Query(..., pattern="^(ssp126|ssp245|ssp370|ssp585)$"),
        start_year: int = Query(..., ge=2015, le=2100),
        end_year: int   = Query(..., ge=2015, le=2100),
        mode: str = Query("baseline", pattern="^(baseline|percentile)$"),
        q: float = Query(0.95, ge=0.0, le=1.0),
        basis: str = Query("ensemble2015"),
        agg: str = Query("median", pattern="^(median|mean)$"),
        stat: str = Query("ratio", pattern="^(ratio|delta)$"),
        bucket: str = Query(DEFAULT_BUCKET),
):
    try:
        client = _minio()
        res = compute_region_shock(
            client=client, bucket=bucket,
            region_iso2=region, scenario=scenario,
            y0=start_year, y1=end_year,
            mode=mode, q=q, basis=basis, agg=agg, stat=stat,
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/threshold_map")
def threshold_map(
        scenario: str = Query(..., pattern="^(ssp126|ssp245|ssp370|ssp585)$"),
        quantile: float = Query(0.95, ge=0.0, le=1.0),
        basis: str = Query("ensemble2015"),
        bucket: str = Query(DEFAULT_BUCKET),
):
    try:
        client = _minio()
        res = build_threshold_map(
            client=client,
            bucket=bucket,
            scenario=scenario,
            quantile=quantile,
            basis=basis,
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WICHTIG: relativer Pfad, weil der Router in main.py bereits prefix="/api/pm25" hat.
@router.get("/files")
def list_index():
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(status_code=404, detail="Index not found. Run POST /api/pm25/files/reindex first.")
    with open(INDEX_PATH, "r") as f:
        return json.load(f)

@router.get("/region")
def region_value(region: str = Query(...), scenario: str = Query(...)):
    value = get_region_value(region.upper(), scenario.lower())
    if value is None:
        raise HTTPException(status_code=404, detail="Region or scenario not found")
    return {"value": value}

@router.get("/threshold")
def threshold(
        scenario: str = Query(..., pattern="^(ssp126|ssp245|ssp370|ssp585)$"),
        quantile: float = Query(0.95, ge=0.0, le=1.0),
        force: bool = Query(False),
        bucket: str = Query(DEFAULT_BUCKET),
):
    try:
        client = _minio()
        res = build_threshold_map(
            client=client, bucket=bucket,
            scenario=scenario, quantile=quantile, basis="ensemble2015",
        )
        import xarray as xr, numpy as np
        ds = xr.open_dataset(res["path"])
        try:
            glob_val = float(np.nanmedian(ds["q"].values))  # oder Flächengewichtung
        finally:
            ds.close()
        return {
            "value": glob_val,
            "source_map": res["path"],
            "n_inputs": res["n_inputs"],
            "n_models": res["n_models"],
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/country_mask/check_grid")
def country_mask_check_grid(
        res_deg: float = Query(1.0, ge=0.1, le=5.0),
        tol: float = Query(1e-6),
        mask_path: Optional[str] = Query(None),
):
    """Prüft: passt die Maskendatei exakt zum erwarteten Zielgrid?"""
    try:
        return check_mask_vs_target(mask_path=mask_path, res_deg=res_deg, tol=tol)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/country_mask/validate")
def country_mask_validate(
        scenario: Optional[str] = Query(None),
        bucket: str = Query(DEFAULT_BUCKET),
        res_deg: float = Query(1.0, ge=0.1, le=5.0),
        tol: float = Query(1e-6),
        mask_path: Optional[str] = Query(None),
):
    """
    Nimmt eine Beispieldatei (optional gefiltert nach scenario),
    regriddet auf Zielgrid und vergleicht Koordinaten mit der Maske.
    """
    try:
        client = _minio()
        return check_alignment_with_sample(
            client=client,
            bucket=bucket,
            scenario=scenario,
            mask_path=mask_path,
            res_deg=res_deg,
            tol=tol,
        )
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))