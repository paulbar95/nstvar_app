from fastapi import FastAPI
from api import pm25
from api.endpoints import indexer


app = FastAPI(title="PM2.5 AII Service")

app.include_router(pm25.router, prefix="/api/pm25", tags=["PM2.5"])
app.include_router(indexer.router, prefix="/api/pm25", tags=["admin"])
