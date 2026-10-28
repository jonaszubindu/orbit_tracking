"""Approximate maximum station baseline with continuous simultaneous visibility.

Spherical Earth and two-body propagation, consistent with orbit_toy.py.
Both stations are freely placeable. This is a grid search, not a certified global optimum.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orbit_toy import RE, station_state, propagate


def unit_vectors(lat, lon):
    a, b = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack(np.broadcast_arrays(np.cos(a)*np.cos(b), np.cos(a)*np.sin(b), np.sin(a)), axis=-1)


def ecef_satellite(state, epoch, t):
    dt = float((t-epoch)/np.timedelta64(1, 's'))
    r, _ = propagate(state[None, :3].copy(), state[None, 3:].copy(), dt, max_step=30.)
    # Inertial->Earth-fixed rotation defined by the same station model.
    x, _ = station_state(t, 0., 0., 0.)
    ang = np.arctan2(x[1], x[0]); c, s = np.cos(ang), np.sin(ang)
    v = r[0]
    return np.array([c*v[0]+s*v[1], -s*v[0]+c*v[1], v[2]])


def elevation(sat_ecef, u, station_alt_km=0.):
    los = sat_ecef - (RE+station_alt_km)*u
    return np.arcsin(np.clip(np.sum(los*u, axis=-1)/np.linalg.norm(los, axis=-1), -1, 1))


def solve_window(sats, latgrid, longrid, min_elev_deg, refine=True):
    lat, lon = np.meshgrid(latgrid, longrid, indexing='ij')
    u = unit_vectors(lat.ravel(), lon.ravel())
    elev_min = np.minimum.reduce([elevation(s, u) for s in sats])
    mask = elev_min >= np.deg2rad(min_elev_deg)
    available = u[mask]
    if len(available) < 2:
        return None
    # Chord length gives the same maximizer as surface great-circle distance.
    ds = pdist(available, 'euclidean')
    k = int(np.argmax(ds)); n = len(available)
    # Convert scipy condensed index to pair.
    i = int(n - 2 - np.floor(np.sqrt(-8*k+4*n*(n-1)-7)/2 - .5))
    j = int(k + i + 1 - n*(n-1)//2 + (n-i)*(n-i-1)//2)
    aa, bb = available[i], available[j]
    def latlon(vec):
        return np.array([np.rad2deg(np.arcsin(vec[2])), np.rad2deg(np.arctan2(vec[1], vec[0]))])
    p0 = np.r_[latlon(aa), latlon(bb)]
    def separation(p):
        x, y = unit_vectors(p[0], p[1]), unit_vectors(p[2], p[3])
        return np.arccos(np.clip(x@y, -1, 1))
    def constraints(p):
        x, y = unit_vectors(p[0], p[1]), unit_vectors(p[2], p[3])
        return np.array([elevation(s, x) - np.deg2rad(min_elev_deg) for s in sats] +
                        [elevation(s, y) - np.deg2rad(min_elev_deg) for s in sats])
    p = p0
    if refine:
        sol = minimize(lambda q: -separation(q), p0, method='SLSQP',
                       bounds=[(-90,90),(-180,180)]*2,
                       constraints={'type':'ineq', 'fun':constraints},
                       options={'maxiter':120, 'ftol':1e-10})
        if np.min(constraints(sol.x)) >= -1e-6 and separation(sol.x) >= separation(p0):
            p = sol.x
    return {'distance_km':float(RE*separation(p)), 'central_angle_deg':float(np.rad2deg(separation(p))),
            'station_a_lat':float(p[0]),'station_a_lon':float(p[1]),
            'station_b_lat':float(p[2]),'station_b_lon':float(p[3]),
            'grid_visible_stations':int(mask.sum())}


def load_orbit(path, index=None):
    with np.load(path) as z:
        epoch = np.datetime64(z['epoch'].item() if z['epoch'].shape == () else z['epoch'])
        if 'reference_state' in z:
            state = z['reference_state']
        elif 'states' in z:
            states = z['states']; state = states[0 if index is None else index]
        else:
            raise ValueError('NPZ needs reference_state or states')
    return epoch, np.asarray(state, float).reshape(6)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--orbit', required=True, help='NPZ with epoch and reference_state or states')
    ap.add_argument('--state-index', type=int, default=None)
    ap.add_argument('--hours', type=float, default=12.)
    ap.add_argument('--window-step-min', type=float, default=30.)
    ap.add_argument('--overlap-min', type=float, default=2.)
    ap.add_argument('--min-elevation', type=float, default=10.)
    ap.add_argument('--grid-deg', type=float, default=3.)
    ap.add_argument('--samples', type=int, default=5, help='Visibility checks throughout overlap')
    ap.add_argument('--output-dir', default='results/handover')
    args=ap.parse_args()
    if args.overlap_min <= 0 or args.samples < 2 or args.grid_deg <= 0: ap.error('Invalid overlap, samples or grid')
    epoch,state=load_orbit(args.orbit,args.state_index)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    lat=np.arange(-90,90.0001,args.grid_deg)
    lon=np.arange(-180,180,args.grid_deg)
    records=[]
    for minute in np.arange(0,args.hours*60-args.overlap_min+1e-6,args.window_step_min):
        ts=[epoch+np.timedelta64(int(round((minute+dt)*60*1000)),'ms') for dt in np.linspace(0,args.overlap_min,args.samples)]
        sats=[ecef_satellite(state,epoch,t) for t in ts]
        result=solve_window(sats,lat,lon,args.min_elevation)
        if result:
            records.append({'start_utc':str(ts[0]),'end_utc':str(ts[-1]),'start_min':float(minute),**result})
    if not records: raise RuntimeError('No overlapping stations found in requested search interval')
    best=max(records,key=lambda d:d['distance_km'])
    with (out/'overlap_windows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader();writer.writerows(records)
    summary={'description':'Approximate maximum surface baseline for two freely placeable ground stations',
             'orbit':args.orbit,'epoch':str(epoch),'overlap_minutes':args.overlap_min,
             'min_elevation_deg':args.min_elevation,'grid_deg':args.grid_deg,
             'visibility_samples_per_window':args.samples,'best':best,
             'caveat':'Coarse window scan and station grid with local refinement. Not a guaranteed global maximum; visibility between sample times not rigorously certified. Two-body spherical-Earth model.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot([r['start_min']/60 for r in records],[r['distance_km'] for r in records],'.-',label='Max station separation (sampled windows)')
    ax.scatter([best['start_min']/60],[best['distance_km']],color='tab:red',s=60,zorder=3,label='Largest found')
    ax.set(xlabel='Overlap window start, hours after orbit epoch',ylabel='Station surface separation (km)',title=f'2-station handover: {args.overlap_min:g} min simultaneous visibility, elevation ≥ {args.min_elevation:g}°')
    ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'overlap_baseline.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
