"""Standalone sensitivity analysis: angular speed versus orbit range and velocity.

No modifications to the main orbit-fitting and search tools.
Geometric instantaneous angular speed (topocentric), no optical exposure model.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from orbit_toy import MU,RE,station_state,orbital_filter


def angular_speed(r,v,rs,vs):
    d=r-rs
    dv=v-vs
    rho=np.linalg.norm(d,axis=-1)
    u=d/rho[...,None]
    transverse=dv-np.sum(dv*u,axis=-1)[...,None]*u
    return np.linalg.norm(transverse,axis=-1)/rho # rad/s


def analyze(out,lat=46.95,lon=7.44,alt=.55,fov_diameter=1.):
    epoch=np.datetime64('2026-10-22T20:00:00')
    rs,vs=station_state(epoch,lat,lon,alt)
    up=rs/np.linalg.norm(rs)
    east=np.cross([0.,0.,1.],up);east/=np.linalg.norm(east)
    rho=np.geomspace(500,60000,500)
    # Three distinct assumed inertial eastward speeds. The station rotates.
    speeds=[2.0,3.5,5.0]
    fig,axs=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    for speed in speeds:
        r=rs+rho[:,None]*up
        v=np.broadcast_to(speed*east,(len(rho),3))
        w=np.rad2deg(angular_speed(r,v,rs,vs))*3600
        valid,_=orbital_filter(r,v)
        axs[0,0].plot(rho,w,label=f'{speed:g} km/s inertial eastward')
        axs[0,0].plot(rho[~valid],w[~valid],'.',markersize=1,alpha=.18)
        axs[0,1].plot(rho,fov_diameter*3600/np.maximum(w,1e-12),label=f'{speed:g} km/s')
    axs[0,0].set(xscale='log',yscale='log',xlabel='Topocentric range [km]',ylabel='Angular speed [arcsec/s]',title='Angular speed vs range at zenith')
    axs[0,1].set(xscale='log',yscale='log',xlabel='Topocentric range [km]',ylabel=f'Time across {fov_diameter:g}° diameter [s]',title='Idealized straight-through FoV crossing')
    axs[0,1].set_ylim(1,1e5)
    axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
    # Holding range and transverse motion fixed: radial velocity has no first-order effect.
    rr=20000.
    radial=np.linspace(-5,5,201)
    for vtan in [1.,2.,3.5]:
        r=np.broadcast_to(rs+rr*up,(len(radial),3))
        v=vs+radial[:,None]*up+vtan*east
        w=np.rad2deg(angular_speed(r,v,rs,vs))*3600
        axs[1,0].plot(radial,w,label=f'Relative transverse {vtan:g} km/s')
    axs[1,0].set(xlabel='Topocentric radial velocity [km/s]',ylabel='Angular speed [arcsec/s]',title='Instantaneous angular speed is independent of radial velocity')
    axs[1,0].legend(fontsize=8)
    # Same observed angular rate: a continuum of possible range / transverse speed.
    rate_arcsec_s=25.
    vt=rho*np.deg2rad(rate_arcsec_s/3600.)
    axs[1,1].plot(rho,vt,label=f'Fixed observed {rate_arcsec_s:g} arcsec/s')
    axs[1,1].axhline(3.5,color='gray',ls='--',label='3.5 km/s reference')
    axs[1,1].set(xscale='log',xlabel='Topocentric range [km]',ylabel='Required relative transverse speed [km/s]',title='Range–transverse-velocity degeneracy')
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:ax.grid(alpha=.25)
    fig.suptitle('Standalone optical angular-speed sensitivity | Bern-like observer | idealized instantaneous geometry',fontsize=13)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=165);plt.close(fig)
    # Numeric example: same angular speed can arise from different range/speed combinations.
    print('Range [km] | relative transverse speed [km/s] for 25 arcsec/s | 1° crossing [s]')
    for r in [1000,5000,20000,40000]:
        print(f'{r:9.0f} | {r*np.deg2rad(rate_arcsec_s/3600):10.4f} | {3600/rate_arcsec_s:10.1f}')
    print('Saved',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='analysis/angular_speed_sensitivity.png')
    p.add_argument('--fov-diameter',type=float,default=1.)
    a=p.parse_args();analyze(a.output,fov_diameter=a.fov_diameter)
