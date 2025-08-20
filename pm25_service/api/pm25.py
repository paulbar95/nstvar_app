from fastapi import APIRouter, Query, HTTPException
from service.region_value import get_region_value
from service.threshold import get_threshold

router = APIRouter()

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
