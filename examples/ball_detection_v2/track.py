"""Causal trial tracker on existing Rust foreground-cluster proposals. Stdlib only."""
import csv
import itertools
import json
import math
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'outputs/ball-detection-v2'
CALIBRATION = json.loads((REPO / 'calibration.json').read_text())['camera_matrix']
FX, FY = CALIBRATION[0][0], CALIBRATION[1][1]
CX, CY = CALIBRATION[0][2], CALIBRATION[1][2]


def candidates(rows, diameter):
    result = []
    for row in rows:
        if not row['cx']:
            continue
        x, y, w, h, n = [float(row[k]) for k in ['ref_x', 'ref_y', 'width', 'height', 'events']]
        depth = FX * diameter / min(w, h)
        if n >= 30 and min(w, h) / max(w, h) >= .25 and .25 < depth < 15 and max(w, h) < 700:
            result.append(dict(x=x, y=y, w=w, h=h, n=n, valid_circle=float(row.get('coverage') or 0)>=9, camera_x=float(row['cx']), camera_y=float(row['cy'])))
    return result


def camera_position(x, y, row):
    ray = [(x - CX) / FX, (y - CY) / FY, 1.]
    # Recorded matrix maps the current camera view into the reference view.
    q = [sum(float(row[f'r{j}{i}']) * ray[j] for j in range(3)) for i in range(3)]
    if q[2] <= 0:
        return math.nan, math.nan
    return FX * q[0] / q[2] + CX, FY * q[1] / q[2] + CY


# ponytail: geometry can match non-ball edges; add appearance checks if false tracks persist.
def track(groups, diameter):
    active = None
    pending = []
    ident = 0
    sizes = deque(maxlen=5)
    for t, rows in groups:
        available = candidates(rows, diameter)
        state = 'lost'
        if active:
            dt = t - active['t']
            px = active['x'] + active['vx'] * dt
            py = active['y'] + active['vy'] * dt
            size = sorted(sizes)[len(sizes) // 2]
            camera_px,camera_py=camera_position(px,py,rows[0])
            camera_ax,camera_ay=camera_position(active['x'],active['y'],rows[0])
            gate = max(25., 1.5 * max(size,active['w'],active['h'])) + .25 * math.hypot(camera_px-camera_ax,camera_py-camera_ay)
            nearby = [c for c in available if math.hypot(c['camera_x'] - camera_px, c['camera_y'] - camera_py) <= gate and .35 <= min(c['w'], c['h']) / size <= 2.5]
            if nearby:
                circles = [c for c in nearby if c['valid_circle']]
                nearby = circles or nearby
                c = max(nearby, key=lambda c: c['n'] / (1. + math.hypot(c['camera_x'] - camera_px, c['camera_y'] - camera_py) / (gate * .5)))
                rx, ry = c['x'] - px, c['y'] - py
                active.update(x=px + .8 * rx, y=py + .8 * ry, vx=active['vx'] + .2 * rx / dt, vy=active['vy'] + .2 * ry / dt, w=c['w'], h=c['h'], last=t, t=t, valid_circle=c['valid_circle'])
                if c['valid_circle']:
                    sizes.append(min(c['w'], c['h']))
                state = 'measured'
            elif t - active['last'] <= .08 + 1e-9:
                active.update(x=px, y=py, t=t)
                state = 'predicted'
            else:
                active = None
                pending = []
        if active is None:
            seeds = [c for c in available if (pending or c['valid_circle']) and min(c['w'], c['h']) / max(c['w'], c['h']) >= .5 and c['n'] / (c['w'] * c['h']) >= .1]
            if pending and t - pending[-1]['t'] > .035:
                pending = []
            if pending:
                last = pending[-1]
                last_x,last_y=camera_position(last['x'],last['y'],rows[0])
                close = [c for c in seeds if math.hypot(c['camera_x'] - last_x, c['camera_y'] - last_y) <= max(25., 2 * min(c['w'], c['h']))]
                c = max(close, key=lambda c: c['n'] / (1. + math.hypot(c['camera_x'] - last_x, c['camera_y'] - last_y) / 25.)) if close else None
            else:
                c = max(seeds, key=lambda c: c['n']) if seeds else None
            if c:
                pending.append(dict(c, t=t))
                pending = pending[-6:]
                if len(pending) == 6 and sum(p['valid_circle'] for p in pending)>=3:
                    dt = t - pending[0]['t']
                    vx, vy = (c['x'] - pending[0]['x']) / dt, (c['y'] - pending[0]['y']) / dt
                    first_x,first_y=camera_position(pending[0]['x'],pending[0]['y'],rows[0])
                    if math.hypot(c['camera_x']-first_x,c['camera_y']-first_y)/dt >= 120:
                        ident += 1
                        active = dict(c, vx=vx, vy=vy, t=t, last=t)
                        sizes.clear()
                        sizes.extend(min(p['w'], p['h']) for p in pending if p['valid_circle'])
                        state = 'measured'
        record = dict(time_s=t, state=state, track_id=ident, outline_supported=bool(active and state=='measured' and active['valid_circle']))
        if active:
            z = FX * diameter / sorted(sizes)[len(sizes) // 2]
            x, y = camera_position(active['x'], active['y'], rows[0])
            if not (math.isfinite(x) and math.isfinite(y) and -10 < x < 1290 and -10 < y < 730):
                active = None
                pending = []
                record['state'] = 'lost'
                record['outline_supported'] = False
            else:
                record.update(ref_x=active['x'], ref_y=active['y'], width_px=active['w'], height_px=active['h'], camera_x_px=x, camera_y_px=y, depth_m=z, camera_x_m=(x - CX) * z / FX, camera_y_m=(y - CY) * z / FY)
        yield record


def self_check():
    def row(x, y=300):
        d = dict(cx=str(x), cy=str(y), ref_x=str(x), ref_y=str(y), width='20', height='20', events='120', coverage='16')
        d.update({f'r{i}{j}': str(int(i == j)) for i in range(3) for j in range(3)})
        return d
    stationary = [(i / 100, [row(400)]) for i in range(20)]
    assert all(r['state'] == 'lost' for r in track(stationary, .07))
    moving = [(i / 100, [row(400 + 5 * i)]) for i in range(12)]
    missing = [(i / 100, [dict(row(0), cx='')]) for i in range(12, 24)]
    records = list(track(moving + missing, .07))
    assert records[5]['state'] == 'measured'
    assert records[12]['state'] == 'predicted' and records[19]['state'] == 'predicted'
    assert records[20]['state'] == 'lost'
    large = list(track(moving, .25))
    assert math.isclose(large[-1]['depth_m'] / records[11]['depth_m'], .25 / .07)

    # The matching gate must remain in camera pixels after a large IMU rotation.
    rotation = [[1.,0.,0.],[0.,.5,math.sqrt(.75)],[0.,-math.sqrt(.75),.5]]
    rotated = []
    for i in range(12):
        x,y=400.,300.+15*i
        d=row(x,y)
        ray=[(x-CX)/FX,(y-CY)/FY,1.]
        q=[sum(rotation[j][k]*ray[k] for k in range(3)) for j in range(3)]
        d.update(ref_x=str(FX*q[0]/q[2]+CX),ref_y=str(FY*q[1]/q[2]+CY))
        d.update({f'r{j}{k}':str(rotation[j][k])for j in range(3)for k in range(3)})
        rotated.append((i/100,[d]))
    assert all(r['state']=='measured' for r in list(track(rotated,.07))[5:])


if __name__ == '__main__':
    self_check()
    summary = []
    fields = ['time_s', 'state', 'track_id', 'ref_x', 'ref_y', 'width_px', 'height_px', 'camera_x_px', 'camera_y_px', 'depth_m', 'camera_x_m', 'camera_y_m', 'outline_supported']
    for path in sorted(ROOT.glob('*-components.csv')):
        name = path.name.removesuffix('-components.csv')
        diameter = .25 if 'gross' in name else .07
        with path.open() as file:
            groups = [(float(t), list(rows)) for t, rows in itertools.groupby(csv.DictReader(file), key=lambda r: r['time_s'])]
        records = list(track(groups, diameter))
        with (ROOT / f'{name}-tracks.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        segments = []
        for ident, rows in itertools.groupby(records, key=lambda r: r['track_id']):
            measured = [r for r in rows if r['state'] == 'measured']
            if measured:
                segments.append(dict(track_id=ident, start=measured[0]['time_s'], end=measured[-1]['time_s'], measured_updates=len(measured)))
        summary.append(dict(recording=name, diameter_m=diameter, updates=len(records), measured=sum(r['state'] == 'measured' for r in records), predicted=sum(r['state'] == 'predicted' for r in records), segments=segments))
        print(name, segments)
    (ROOT / 'tracking-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
