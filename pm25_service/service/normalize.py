import numpy as np
import xarray as xr

def pick_pm_var(ds: xr.Dataset) -> str:
    """Findet die PM2.5-Variable robust (mmrpm2p5 / pm2p5 / pm25...)."""
    cands = [k for k in ds.data_vars if "pm2" in k.lower() or "pm25" in k.lower()]
    return cands[0] if cands else list(ds.data_vars)[0]

def normalize_da(da: xr.DataArray) -> xr.DataArray:
    """Bringt ein Feld sicher in ("time","lat","lon")."""
    rn = {}
    if "latitude" in da.dims:  rn["latitude"] = "lat"
    if "longitude" in da.dims: rn["longitude"] = "lon"
    if "nav_lat" in da.dims:   rn["nav_lat"]  = "lat"
    if "nav_lon" in da.dims:   rn["nav_lon"]  = "lon"
    if "rlat" in da.dims:      rn["rlat"]     = "lat"
    if "rlon" in da.dims:      rn["rlon"]     = "lon"
    if "y" in da.dims and "lat" not in da.dims: rn["y"] = "lat"
    if "x" in da.dims and "lon" not in da.dims: rn["x"] = "lon"
    if rn: da = da.rename(rn)

    # Vertikalebene (falls vorhanden) auf bodennah reduzieren
    for lvl in ("lev","plev","level","height","altitude"):
        if lvl in da.dims:
            coord = da.coords.get(lvl)
            idx = int(xr.DataArray(coord).argmax().item()) if (coord is not None and np.issubdtype(coord.dtype, np.number)) else -1
            da = da.isel({lvl: idx})
            break

    # Lon in [-180,180) und sortieren
    if "lon" in da.dims:
        lon = da["lon"]
        if float(lon.max()) > 180.0 or float(lon.min()) >= 0.0:
            da = da.assign_coords(lon=((lon + 180.0) % 360.0) - 180.0)
        da = da.sortby("lon")

    # Lat aufsteigend
    if "lat" in da.dims:
        try:
            if not da["lat"].is_monotonic_increasing:
                da = da.sortby("lat")
        except Exception:
            pass

    # Dummy-Time
    if "time" not in da.dims:
        da = da.expand_dims(time=[0])

    return da.transpose("time","lat","lon", missing_dims="ignore")
