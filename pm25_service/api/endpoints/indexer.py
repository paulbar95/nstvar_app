from fastapi import APIRouter
from service.indexer import index_pm25_data

router = APIRouter()

@router.get("/index", tags=["admin"])
def index_data():
    # Diese Werte aus ENV oder config holen!
    BUCKET = "pm25-data"
    ENDPOINT = "http://minio:9000"
    ACCESS_KEY = "minioadmin"
    SECRET_KEY = "minioadmin"
    result = index_pm25_data(BUCKET, minio_endpoint=ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY)
    return {"files": result}
