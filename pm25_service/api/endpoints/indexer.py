import os
from fastapi import APIRouter, Query
from pydantic import BaseModel
from minio import Minio
from service.indexer import index_pm25_data

DEFAULT_BUCKET = os.getenv("PM25_BUCKET", "pm25data")  # <— jetzt auf ENV

router = APIRouter(tags=["admin"])

class ReindexResponse(BaseModel):
    bucket: str
    prefix: str | None
    count: int

def _minio() -> Minio:
    return Minio(
        os.environ.get("MINIO_URL", "minio:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )

@router.post("/files/reindex", response_model=ReindexResponse)
def reindex(bucket: str = DEFAULT_BUCKET, prefix: str | None = Query(None)):
    client = _minio()
    cnt = index_pm25_data(client, bucket=bucket, prefix=prefix or "")
    return {"bucket": bucket, "prefix": prefix, "count": cnt}
