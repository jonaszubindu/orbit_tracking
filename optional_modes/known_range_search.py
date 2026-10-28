"""Additional independent search entry point: known state or approximate range.

Uses existing sky_search_prototype without changing the legacy pipelines.
Known state is six ECI-like Cartesian components at --epoch.
Range conditioning is a Gaussian likelihood at --epoch relative to a ranging station.
It reweights an EXISTING ensemble, so it cannot create missing orbit families.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from orbit_toy import station_state, propagate
from orbit_station_search import Station
from sky_search_prototype.sky_search_maps import generate


def build_known(state, epoch):
    state=np.asarray(state,dtype=float)
    if state.shape!=(6,) or not np.isfinite(state).all():
        raise ValueError('Known state must contain six finite ECI-like components')
    return dict(states=state[None,:],weights=np.ones(1),family_ids=np.zeros(1,int),epoch=str(np.datetime64(epoch,'ns')))


def condition_range(path, range_km, sigma_km, station, max_samples=None, measurement_epoch=None):
    if sigma_km<=0 or range_km<=0:
        raise ValueError('Range and range standard deviation must be positive')
    with np.load(path,allow_pickle=False) as data:
        states=np.asarray(data['states'],float)
        epoch=np.datetime64(str(data['epoch']),'ns')
        weights=np.asarray(data['weights'],float) if 'weights' in data else np.ones(len(states))
        family=np.asarray(data['family_ids'],int) if 'family_ids' in data else np.zeros(len(states),int)
    if states.ndim!=2 or states.shape[1]!=6 or len(states)==0 or not np.isfinite(states).all():
        raise ValueError('Invalid input states')
    if weights.shape!=(len(states),) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Invalid prior ensemble weights')
    if max_samples and len(states)>max_samples:
        raise ValueError('Too many input states: reduce source ensemble beforehand, preserving weights')
    observed_epoch = np.datetime64(measurement_epoch, 'ns') if measurement_epoch is not None else epoch
    delta_s = float((observed_epoch - epoch) / np.timedelta64(1, 's'))
    if abs(delta_s) > 7 * 86400:
        raise ValueError('Range observation more than 7 days from ensemble epoch; toy two-body model is unsuitable')
    if delta_s:
        positions, _ = propagate(states[:, :3].copy(), states[:, 3:].copy(), delta_s)
    else:
        positions = states[:, :3]
    station_r,_=station_state(observed_epoch,station.lat_deg,station.lon_deg,station.altitude_km)
    rho=np.linalg.norm(positions-station_r,axis=1)
    logw=np.log(np.maximum(weights/weights.sum(),1e-300))-.5*((rho-range_km)/sigma_km)**2
    logw-=logw.max()
    w=np.exp(logw);w/=w.sum()
    ess=1/np.sum(w*w)
    return dict(states=states,weights=w,family_ids=family,epoch=str(epoch),range_measurement_epoch=str(observed_epoch),range_measurement_km=float(range_km),range_sigma_km=float(sigma_km)),rho,ess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['known','range'],required=True)
    p.add_argument('--state',nargs=6,type=float,help='Known [rx ry rz vx vy vz] at --epoch (km, km/s)')
    p.add_argument('--epoch',help='UTC state epoch for known mode')
    p.add_argument('--input',help='Existing single MC or global Stage B NPZ for range mode')
    p.add_argument('--range-km',type=float,help='Measured topocentric range at input state epoch')
    p.add_argument('--range-sigma-km',type=float,help='1-sigma range uncertainty')
    p.add_argument('--range-epoch',help='UTC time of range observation, e.g. 2026-10-22T20:10:00; defaults to ensemble epoch')
    p.add_argument('--range-lat',type=float,default=46.95)
    p.add_argument('--range-lon',type=float,default=7.44)
    p.add_argument('--range-alt-km',type=float,default=.55)
    p.add_argument('--lat',type=float,required=True)
    p.add_argument('--lon',type=float,required=True)
    p.add_argument('--alt-km',type=float,default=0.)
    p.add_argument('--name',default='Custom station')
    p.add_argument('--hours',type=float,default=12.)
    p.add_argument('--step-minutes',type=float,default=20.)
    p.add_argument('--min-elevation',type=float,default=10.)
    p.add_argument('--fov-radius',type=float,default=1.)
    p.add_argument('--map-radius',type=float,default=12.)
    p.add_argument('--output',default='results/known_or_range_search.png')
    a=p.parse_args()
    if a.mode=='known':
        if a.state is None or not a.epoch: p.error('--state and --epoch required for known mode')
        payload=build_known(a.state,a.epoch)
        print('KNOWN ORBIT: one exact assumed state; coverage is 0 or 100%, not detection probability')
    else:
        if not a.input or a.range_km is None or a.range_sigma_km is None:
            p.error('--input, --range-km, --range-sigma-km required for range mode')
        station=Station('Ranging station',a.range_lat,a.range_lon,a.range_alt_km)
        payload,rho,ess=condition_range(a.input,a.range_km,a.range_sigma_km,station,measurement_epoch=a.range_epoch)
        print(f'RANGE CONDITIONING: posterior ESS={ess:.1f}/{len(rho)}, measured range {a.range_km:g}±{a.range_sigma_km:g} km')
        if ess<30:print('WARNING: severe weight collapse; increase source sample coverage or resample the posterior')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    intermediate=out.with_name(out.stem+'_conditioned.npz')
    np.savez_compressed(intermediate,**payload)
    generate(intermediate,'global',Station(a.name,a.lat,a.lon,a.alt_km),out,a.hours,a.step_minutes,a.min_elevation,a.fov_radius,a.map_radius)
    print('Saved conditioned ensemble:',intermediate)

if __name__=='__main__':main()
