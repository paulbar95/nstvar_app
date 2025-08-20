import boto3
import os
import re
import json

def index_pm25_data(bucket_name, prefix="", minio_endpoint=None, access_key=None, secret_key=None):
    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    files = response.get('Contents', [])
    index = []
    # REGEX pattern wie oben
    pattern = (
        r"(?P<var>[^_]+)_"
        r"(?P<freq>[^_]+)_"
        r"(?P<model>[^_]+)_"
        r"(?P<scenario>[^_]+)_"
        r"(?P<run>[^_]+)_"
        r"(?P<grid>[^_]+)_"
        r"(?P<start>\d{6})-(?P<end>\d{6})\.nc"
    )
    for f in files:
        filename = f['Key']
        m = re.match(pattern, os.path.basename(filename))
        if m:
            entry = m.groupdict()
            entry['file'] = filename
            index.append(entry)
    return index
