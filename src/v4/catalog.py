"""Download the SCEDC earthquake catalogue for the Ridgecrest DAS period.

Three tables are written to data/v4/:
  catalog_events.csv   M >= min_magnitude within radius_deg (candidate signals)
  catalog_screen.csv   every catalogued event within screen_radius_deg (noise screening)
  catalog_regional.csv M >= regional_magnitude within regional_radius_deg (noise screening)
"""

import pandas as pd

from .das import CFG, DATA, geometry, haversine_km

KM_PER_DEG = 111.19


def to_frame(catalog):
    rows = []
    for ev in catalog:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude() or (ev.magnitudes[0] if ev.magnitudes else None)
        rows.append(dict(
            event_id=str(ev.resource_id).replace("=", "/").rstrip("/").rsplit("/", 1)[-1],
            time=pd.Timestamp(o.time.datetime, tz="UTC"),
            latitude=o.latitude, longitude=o.longitude,
            depth_km=(o.depth or 0.0) / 1000.0,
            magnitude=m.mag if m else float("nan"),
            magnitude_type=m.magnitude_type if m else "",
        ))
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def within(d, lat, lon, radius_deg):
    """Keep events whose epicentre lies within radius_deg of (lat, lon).

    Applied after every query: the smoke test showed the service returning
    events thousands of km away despite maxradius.
    """
    if d.empty:
        return d
    km = haversine_km(lat, lon, d.latitude.values, d.longitude.values)
    return d[km <= radius_deg * KM_PER_DEG].reset_index(drop=True)


def query(client, lat, lon, radius, minmag, start, end):
    """Query day by day so no single request hits a server row limit."""
    from obspy.clients.fdsn.header import FDSNNoDataException

    frames = []
    for day in pd.date_range(start, end, freq="D", tz="UTC"):
        try:
            cat = client.get_events(
                starttime=day.isoformat(), endtime=(day + pd.Timedelta(days=1)).isoformat(),
                latitude=lat, longitude=lon, maxradius=radius, minmagnitude=minmag)
        except FDSNNoDataException:
            continue
        frames.append(to_frame(cat))
    d = pd.concat(frames, ignore_index=True).drop_duplicates("event_id")
    received = len(d)
    d = within(d, lat, lon, radius)
    if minmag is not None:
        d = d[d.magnitude >= minmag]
    print(f"  received {received}, kept {len(d)} within {radius} deg")
    return d.sort_values("time").reset_index(drop=True)


def main():
    from obspy.clients.fdsn import Client

    client = Client("SCEDC")
    c, a = CFG["catalog"], CFG["archive"]
    lat, lon, _ = geometry()
    start, end = a["first_day"], a["last_day"]
    DATA.mkdir(parents=True, exist_ok=True)
    tables = {
        "catalog_events.csv": (lat, lon, c["radius_deg"], c["min_magnitude"]),
        "catalog_screen.csv": (lat, lon, c["screen_radius_deg"], None),
        "catalog_regional.csv": (lat, lon, c["regional_radius_deg"], c["regional_magnitude"]),
    }
    for name, args in tables.items():
        d = query(client, *args, start, end)
        d.to_csv(DATA / name, index=False)
        print(f"{name}: {len(d)} events")


if __name__ == "__main__":
    main()
