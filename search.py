"""Quick station search: default single ensemble, optional approximate-range conditioning.

Run `python search.py --help`. Range is topocentric from --range-lat/--range-lon
at the ensemble state epoch. This is not geocentric altitude or slant range from
future search station. Probability interpretations depend on input ensemble.
"""
import argparse
from pathlib import Path
import sys
import numpy as np

from orbit_station_search import Station
from sky_search_prototype.sky_search_maps import generate
from optional_modes.known_range_search import condition_range


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',choices=['single','global'],default='single',help='Source orbit ensemble; omitted means single-family mode')
    p.add_argument('--input',help='Input NPZ. Defaults to bundled example for chosen mode')
    p.add_argument('--lat',type=float,required=True,help='Search station latitude [deg]')
    p.add_argument('--lon',type=float,required=True,help='Search station longitude [deg]')
    p.add_argument('--alt-km',type=float,default=0,help='Search station altitude [km]')
    p.add_argument('--name',default='Custom station')
    p.add_argument('--range-km',type=float,help='Optional topocentric range measured at input epoch [km]')
    p.add_argument('--range-sigma-km',type=float,help='1-sigma uncertainty in range [km]')
    p.add_argument('--range-epoch',help='UTC time of independent range measurement (ISO 8601); defaults to ensemble epoch')
    p.add_argument('--range-lat',type=float,default=46.95,help='Ranging station latitude [deg] (Bern default)')
    p.add_argument('--range-lon',type=float,default=7.44,help='Ranging station longitude [deg] (Bern default)')
    p.add_argument('--range-alt-km',type=float,default=0.55,help='Ranging station altitude [km]')
    p.add_argument('--hours',type=float,default=16)
    p.add_argument('--step-minutes',type=float,default=20)
    p.add_argument('--min-elevation',type=float,default=10)
    p.add_argument('--fov-radius',type=float,default=1)
    p.add_argument('--map-radius',type=float,default=12)
    p.add_argument('--output',default='results/quick_search.png')
    a=p.parse_args(argv)
    if (a.range_km is None)!=(a.range_sigma_km is None):
        p.error('Supply both --range-km and --range-sigma-km, or neither')
    if a.range_km is not None and (a.range_km<=0 or a.range_sigma_km<=0):
        p.error('Range and range uncertainty must be positive')
    if not (-90<=a.lat<=90 and -90<=a.range_lat<=90 and -180<=a.lon<=180 and -180<=a.range_lon<=180):
        p.error('Invalid station latitude/longitude')
    if a.hours<=0 or a.step_minutes<=0 or a.fov_radius<=0 or a.map_radius<=0:
        p.error('Hours, step size, FoV radius and map radius must be positive')
    root=Path(__file__).resolve().parent
    default='mc_fitted_5_min.npz' if a.mode=='single' else 'stage_b_3_min.npz'
    source=Path(a.input) if a.input else root/'examples'/default
    if not source.is_file():p.error(f'Input NPZ not found: {source}')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    if a.range_km is None:
        print(f'NO RANGE PROVIDED: using {a.mode} ensemble unchanged')
        path=source
        render_mode=a.mode
    else:
        station=Station('Ranging station',a.range_lat,a.range_lon,a.range_alt_km)
        payload,ranges,ess=condition_range(source,a.range_km,a.range_sigma_km,station,measurement_epoch=a.range_epoch)
        conditioned=output.with_name(output.stem+'_conditioned.npz')
        np.savez_compressed(conditioned,**payload)
        path=conditioned
        # 'global' loader preserves weights. This does NOT imply global-posterior calibration.
        render_mode='global'
        print(f'RANGE UPDATE: {a.range_km:g} ± {a.range_sigma_km:g} km from {station.lat_deg:g}°, {station.lon_deg:g}° at {payload["range_measurement_epoch"]}')
        print(f'Weighted ESS: {ess:.1f} / {len(ranges)}')
        if ess<30 or ess/len(ranges)<0.01:
            print('WARNING: LOW ESS. Search map may be dominated by a few samples; conditional probabilities are unreliable.')
        if a.mode=='single':
            print('Single-family weights are conditional candidate weights, NOT calibrated posterior probabilities.')
    generate(path,render_mode,Station(a.name,a.lat,a.lon,a.alt_km),output,a.hours,a.step_minutes,a.min_elevation,a.fov_radius,a.map_radius)
    print('Time-series CSV:',output.with_suffix('.csv'))
    return 0

if __name__=='__main__':sys.exit(main())
