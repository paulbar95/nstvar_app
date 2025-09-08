import os
import re
import json
from typing import Optional, Dict, Any, List
from minio import Minio

INDEX_PATH = os.environ.get("PM25_INDEX_PATH", "/app/cache/pm25_index.json")

_KEY_RE = re.compile(
    r'^(?P<var>[^_]+)_(?P<freq>[^_]+)_(?P<model>[^_]+)_(?P<scenario>[^_]+)'
    r'_(?P<run>[^_]+)_(?P<grid>[^_]+)_(?P<start>\d{6})-(?P<end>\d{6})\.nc$'
)

def _parse_key(key: str) -> Optional[Dict[str, Any]]:
    name = key.split("/")[-1]
    m = _KEY_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["key"] = key
    return d

def index_pm25_data(client: Minio, bucket: str, prefix: str = "") -> int:
    """Scannt MinIO und schreibt (optional) einen JSON-Index. Gibt die Trefferzahl zurück."""
    items: List[Dict[str, Any]] = []
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        k = obj.object_name
        if not k.endswith(".nc"):
            continue
        parsed = _parse_key(k)
        if parsed:
            items.append(parsed)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"files": items, "count": len(items)}, f)
    return len(items)
