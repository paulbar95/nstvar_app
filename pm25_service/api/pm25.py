from fastapi import APIRouter, Query, HTTPException
import os, json  # <— fehlte
from service.region_value import get_region_value
from service.threshold import get_threshold

# falls du region_shock sofort nutzen willst:
from service.indexer import index_pm25_data         # <— damit kein NameError
# from service.region_shock import compute_region_shock  # <— sobald vorhanden

router = APIRouter(tags=["PM2.5"])

INDEX_PATH = "/app/cache/pm25_index.json"

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
def threshold(scenario: str = Query(...)):
    value = get_threshold(scenario.lower())
    if value is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {"value": value}

@router.get("/region_shock")
def region_shock(
        region: str = Query(..., min_length=2, max_length=2, description="ISO-3166-1 alpha-2"),
        scenario: str = Query(..., pattern="^(ssp126|ssp245|ssp370|ssp585)$"),
        start_year: int = Query(..., ge=2015, le=2100),
        end_year: int = Query(..., ge=2015, le=2100),
        prefix: str = "",
):
    # PLACEHOLDER bis compute_region_shock implementiert ist:
    # Entferne die nächste Zeile und nutze deine echte Implementierung, sobald vorhanden.
    raise HTTPException(status_code=501, detail="region_shock not implemented yet")

    # — echte Variante (wenn compute_region_shock verfügbar) —
    # files = index_pm25_data(client_or_none, bucket_name=None, prefix=prefix or "", ...)
    # res = compute_region_shock(files, alpha2=region, scenario=scenario, y0=start_year, y1=end_year)
    # return res
