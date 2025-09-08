import os, s3fs, xarray as xr

def _storage_opts():
    return dict(
        key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
        client_kwargs={"endpoint_url": os.getenv("MINIO_ENDPOINT", "http://minio:9000")},
    )

def open_dataset_s3(key: str) -> xr.Dataset:
    """key = Pfad innerhalb des Buckets, z.B. 'mmrpm2p5_AERmon_...nc'."""
    bucket = os.getenv("MINIO_BUCKET", "pm25data")
    fs = s3fs.S3FileSystem(**_storage_opts())
    url = f"s3://{bucket}/{key}"
    # Engine automatisch, die CMIP6-Dateien sind meist netCDF4
    return xr.open_dataset(fs.open(url, mode="rb"))