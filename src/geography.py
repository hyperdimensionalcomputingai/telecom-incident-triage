"""The public streets provide a setting, never a coverage or performance model."""
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from config import ROOT, file_hash, save_json

GEOGRAPHY = ROOT / 'data' / 'geography'
ATTRIBUTION = 'Contains information licensed under the Open Government Licence – Toronto.'


def refresh_snapshot(root=GEOGRAPHY):
    """Explicit opt-in refresh; ordinary study preparation uses the frozen snapshot."""
    root = Path(root)
    url = 'https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/datastore_search?' + urllib.parse.urlencode({
        'resource_id': 'ad296ebf-fca6-4e67-b3ce-48040a20e6cd',
        'filters': json.dumps({'LINEAR_NAME_FULL': 'Bloor St W'}),
        'limit': 1000, 'sort': 'CENTRELINE_ID asc'})
    raw = urllib.request.urlopen(url, timeout=60).read()
    result = json.loads(raw)['result']
    if len(result['records']) != result['total']:
        raise ValueError('Snapshot response was truncated')
    features = []
    for row in result['records']:
        geometry = json.loads(row['geometry'])
        points = geometry['coordinates'] if geometry['type'] == 'LineString' else [p for line in geometry['coordinates'] for p in line]
        if any(-79.48 <= p[0] <= -79.39 and 43.65 <= p[1] <= 43.68 for p in points):
            features.append({'type': 'Feature', 'id': row['CENTRELINE_ID'],
                             'properties': {'centreline_id': row['CENTRELINE_ID'], 'street': row['LINEAR_NAME_FULL']},
                             'geometry': geometry})
    if not features:
        raise ValueError('No public corridor geometry')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'bloor-source-response.json').write_bytes(raw)
    save_json(root / 'bloor-corridor.geojson', {'type': 'FeatureCollection', 'features': features})
    save_json(root / 'manifest.json', {
        'dataset': 'Toronto Centreline (TCL)',
        'dataset_url': 'https://ckan0.cf.opendata.inter.prod-toronto.ca/en/dataset/toronto-centreline-tcl',
        'request_url': url, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'coordinate_system': 'EPSG:4326; longitude, latitude; angular coordinates only; no distance computation',
        'selection': 'Bloor St W segments intersecting longitude [-79.48,-79.39], latitude [43.65,43.68]',
        'source_records': len(result['records']), 'corridor_segments': len(features),
        'source_sha256': file_hash(root / 'bloor-source-response.json'),
        'corridor_sha256': file_hash(root / 'bloor-corridor.geojson'),
        'licence_url': 'https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-licence/',
        'attribution': ATTRIBUTION,
        'telecom_data': 'All subscribers, phone trajectories, cells, links, topology and telemetry are simulated. Public streets do not describe network coverage.'})


def load_geography(root=GEOGRAPHY):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, key in (('bloor-source-response.json', 'source_sha256'), ('bloor-corridor.geojson', 'corridor_sha256')):
        if file_hash(root / name) != manifest[key]:
            raise ValueError(f'Public snapshot checksum mismatch: {name}')
    data = json.loads((root / 'bloor-corridor.geojson').read_text())
    lines = []
    for feature in data['features']:
        g = feature['geometry']
        lines.extend([g['coordinates']] if g['type'] == 'LineString' else g['coordinates'])
    lines = [line for line in lines if len(line) >= 2]
    if not lines:
        raise ValueError('Corridor has no valid lines')
    return lines, manifest


def route_points(lines, rng):
    """Three locations along one public road segment; no RF propagation inference."""
    line = lines[int(rng.integers(len(lines)))]
    lengths = [math.dist(a, b) for a, b in zip(line, line[1:])]
    total = sum(lengths)
    if total <= 0:
        raise ValueError('Degenerate public line')
    points = []
    for fraction in (0.15, 0.5, 0.85):
        target = total * fraction
        distance = 0.0
        for a, b, length in zip(line, line[1:], lengths):
            if length and distance + length >= target:
                t = (target - distance) / length
                points.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
                break
            distance += length
    return points


def polygon_for(episode):
    xs = [o['longitude'] for o in episode['observations']]
    ys = [o['latitude'] for o in episode['observations']]
    lo, hi = min(xs) - .012, max(xs) + .012
    bottom, top = min(ys) - .004, max(ys) + .004
    return f'POLYGON (({lo} {bottom}, {hi} {bottom}, {hi} {top}, {lo} {top}, {lo} {bottom}))'
