"""Predict a second station's sky search area from local admissible orbit samples.

Coordinates use the same approximate ECI / rotating spherical Earth model as orbit_toy.
Probabilities are *empirical fractions of accepted local proposal samples*, not a
calibrated posterior or an optical detection probability.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from orbit_toy import MU, station_state

@dataclass(frozen=True)
class Station:
    name: str
    lat_deg: float
    lon_deg: float
    altitude_km: float = 0.0

STATIONS = {
    'Toulouse': Station('Toulouse, France', 43.6045, 1.4440, 0.15),
    'La_Silla': Station('La Silla Observatory, Chile', -29.2567, -70.7346, 2.4),
}

def _propagate_ensemble(states, seconds, max_step_s=180.):
    """Vectorized two-body RK4. State epoch is common to all orbits.

    Efficiently steps forward from epoch; seconds must be nonnegative and sorted.
    """
    t=np.asarray(seconds,float)
    if np.any(t<0) or np.any(np.diff(t)<0):
        raise ValueError('Prediction seconds must be nonnegative and sorted')
    s=np.asarray(states,float).copy()
    r=s[:,:3].copy();v=s[:,3:].copy()
    out=np.empty((len(t),len(s),3))
    def acc(x):
        d=np.linalg.norm(x,axis=1)
        return -MU*x/np.maximum(d,1.)[:,None]**3
    prev=0.
    for i,target in enumerate(t):
        span=target-prev
        n=max(1,int(np.ceil(span/max_step_s)))
        h=span/n
        for _ in range(n):
            k1r=v; k1v=acc(r)
            k2r=v+.5*h*k1v; k2v=acc(r+.5*h*k1r)
            k3r=v+.5*h*k2v; k3v=acc(r+.5*h*k2r)
            k4r=v+h*k3v; k4v=acc(r+h*k3r)
            r=r+(h/6.)*(k1r+2*k2r+2*k3r+k4r)
            v=v+(h/6.)*(k1v+2*k2v+2*k3v+k4v)
        out[i]=r
        prev=target
    return out

def _angles_and_elevation(position, stations):
    los=position-stations[:,None,:]
    ranges=np.linalg.norm(los,axis=2)
    unit=los/ranges[:,:,None]
    ra=np.rad2deg(np.arctan2(unit[:,:,1],unit[:,:,0]))%360
    dec=np.rad2deg(np.arcsin(np.clip(unit[:,:,2],-1,1)))
    up=stations/np.linalg.norm(stations,axis=1)[:,None]
    elev=np.rad2deg(np.arcsin(np.clip(np.einsum('tnk,tk->tn',unit,up),-1,1)))
    return ra,dec,elev,unit

def _great_circle_deg(a,b):
    return np.rad2deg(np.arccos(np.clip(np.sum(a*b,axis=-1),-1,1)))

@dataclass
class SearchPrediction:
    station: Station
    epoch: np.datetime64
    times: np.ndarray
    ra_deg: np.ndarray           # [time, candidate]
    dec_deg: np.ndarray
    elevation_deg: np.ndarray
    reference_ra_deg: np.ndarray # [time]
    reference_dec_deg: np.ndarray
    reference_elevation_deg: np.ndarray
    visible_fraction: np.ndarray
    fov_fraction: np.ndarray     # fraction of all candidates inside reference-centered FoV and above min elevation
    conditional_fov_fraction: np.ndarray  # fraction of visible candidates inside reference-centered FoV
    has_pass: bool
    reference_has_pass: bool
    min_elevation_deg: float
    fov_radius_deg: float

    def best_times(self, n=5):
        ids=np.flatnonzero(self.visible_fraction>0)
        order=sorted(ids,key=lambda i:(-self.fov_fraction[i],-self.visible_fraction[i],i))
        return order[:n]


def predict_station_search(states, reference_state, epoch, station, *,
                           hours=24., step_minutes=10., min_elevation_deg=10.,
                           fov_radius_deg=1., max_step_s=180.):
    """Propagate candidate orbits and compute time-dependent station visibility.

    fov_fraction uses a *single* circular FoV centered on the deterministic
    reference orbit at each time. For multimodal clouds this is not an optimal
    telescope pointing. Visibility is purely geometric; no Sun/illumination/weather.
    """
    if hours<=0 or step_minutes<=0 or fov_radius_deg<=0:
        raise ValueError('hours, step_minutes and fov_radius_deg must be positive')
    states=np.asarray(states,float)
    if states.ndim!=2 or states.shape[1]!=6 or not len(states):
        raise ValueError('states must be a nonempty (N,6) array')
    ref=np.asarray(reference_state,float)
    if ref.shape!=(6,):raise ValueError('reference_state must be length 6')
    epoch=np.datetime64(epoch,'ns')
    step=step_minutes*60.
    seconds=np.arange(0.,hours*3600.+step*.01,step)
    if seconds[-1]<hours*3600.-1e-6:seconds=np.r_[seconds,hours*3600.]
    times=epoch+np.rint(seconds*1e9).astype('timedelta64[ns]')
    all_states=np.vstack((states,ref[None,:]))
    xyz=_propagate_ensemble(all_states,seconds,max_step_s=max_step_s)
    st=np.array([station_state(t,station.lat_deg,station.lon_deg,station.altitude_km)[0] for t in times])
    ra,dec,el,unit=_angles_and_elevation(xyz,st)
    ref_unit=unit[:,-1,:]
    sep=_great_circle_deg(unit[:,:-1,:],ref_unit[:,None,:])
    visible=el[:,:-1]>=min_elevation_deg
    in_fov=visible&(sep<=fov_radius_deg)
    n_visible=visible.sum(axis=1)
    fraction=visible.mean(axis=1)
    fov=in_fov.mean(axis=1)
    conditional=np.divide(in_fov.sum(axis=1),n_visible,
                          out=np.zeros(len(seconds)),where=n_visible>0)
    return SearchPrediction(station,epoch,times,ra[:,:-1],dec[:,:-1],el[:,:-1],
                            ra[:,-1],dec[:,-1],el[:,-1],fraction,fov,conditional,
                            bool(np.any(visible)),bool(np.any(el[:,-1]>=min_elevation_deg)),
                            min_elevation_deg,fov_radius_deg)

def plot_station_search(pred, *,output=None, representative_index=None):
    """3-panel station search summary with an on-sky probability density."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import MaxNLocator
    t=np.asarray((pred.times-pred.epoch)/np.timedelta64(1,'h'),float)
    fig=plt.figure(figsize=(13,9),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=(1.,1.15))
    ax=fig.add_subplot(gs[0,0]); bx=fig.add_subplot(gs[0,1]); sky=fig.add_subplot(gs[1,:])
    ax.plot(t,pred.visible_fraction,label='Above elevation limit',color='#2463a0',lw=2)
    ax.plot(t,pred.fov_fraction,label=f'In reference-centered {pred.fov_radius_deg:g}° FoV',color='#d38325',lw=2)
    ax.set(xlabel='Hours since fitted state epoch',ylabel='Fraction of accepted candidates',ylim=(-.03,1.03))
    ax.legend(fontsize=9);ax.grid(alpha=.25)
    bx.plot(t,pred.reference_elevation_deg,label='Reference orbit',color='black',lw=1.8)
    lo=np.percentile(pred.elevation_deg,[10,90],axis=1)
    bx.fill_between(t,lo[0],lo[1],color='#7caad0',alpha=.4,label='10–90% of candidates')
    bx.axhline(pred.min_elevation_deg,color='#d05a45',ls='--',label='Elevation limit')
    bx.set(xlabel='Hours since fitted state epoch',ylabel='Geometric elevation [deg]')
    bx.grid(alpha=.25);bx.legend(fontsize=9)
    if representative_index is None:
        viable=np.flatnonzero(pred.visible_fraction>0)
        if len(viable):
            representative_index=int(viable[np.argmax(pred.fov_fraction[viable])])
        else: representative_index=int(np.argmax(pred.reference_elevation_deg))
    i=representative_index
    visible=pred.elevation_deg[i]>=pred.min_elevation_deg
    # Tangent-plane offsets around the reference direction, not absolute RA bins.
    center_ra=pred.reference_ra_deg[i]; center_dec=pred.reference_dec_deg[i]
    dra=((pred.ra_deg[i]-center_ra+180)%360-180)*np.cos(np.deg2rad(center_dec))
    ddec=pred.dec_deg[i]-center_dec
    if np.any(visible):
        x=dra[visible];y=ddec[visible]
        lim=np.maximum(np.percentile(np.abs(np.r_[x,y]),98)*1.2,pred.fov_radius_deg*1.4)
        lim=min(max(lim,0.05),180.)
        h,xe,ye=np.histogram2d(x,y,bins=65,range=[[-lim,lim],[-lim,lim]])
        h=gaussian_filter(h,1.15)
        sky.pcolormesh(xe,ye,h.T,cmap='Blues',shading='auto')
        sky.scatter(x,y,s=9,alpha=.18,color='#25538a',label='Visible candidates')
    else:
        lim=2*pred.fov_radius_deg
        sky.text(.5,.5,'No candidate visible at this time',transform=sky.transAxes,ha='center')
    theta=np.linspace(0,2*np.pi,300)
    sky.plot(pred.fov_radius_deg*np.cos(theta),pred.fov_radius_deg*np.sin(theta),
             color='#e07022',lw=2,label='Reference-centered FoV')
    sky.scatter([0],[0],marker='+',s=140,color='red',label='Reference prediction',zorder=10)
    sky.set(xlabel=r'RA offset × cos(Dec) [deg]',ylabel='Declination offset [deg]',
            title=f'On-sky search area at {str(pred.times[i])[:16]} UTC | '
                  f'visible {pred.visible_fraction[i]:.1%}, FoV {pred.fov_fraction[i]:.1%}')
    sky.set_aspect('equal',adjustable='box');sky.set_xlim(-lim,lim);sky.set_ylim(-lim,lim)
    sky.grid(alpha=.2);sky.legend(fontsize=9,loc='upper right')
    flag=('PASS POSSIBLE' if pred.has_pass else 'NO PASS ABOVE ELEVATION LIMIT')
    ref_flag=('reference visible at some time' if pred.reference_has_pass else 'reference never visible')
    fig.suptitle(f'{pred.station.name}: {flag}\n{ref_flag}; ensemble-based fractions (not calibrated detection probabilities)',fontsize=14)
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=160)
    return fig

def save_search_csv(pred,path):
    import csv
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['utc','hours_since_epoch','visible_fraction','fov_fraction_all',
                    'fov_fraction_given_visible','reference_ra_deg','reference_dec_deg',
                    'reference_elevation_deg','any_candidate_above_limit','reference_above_limit'])
        for j,ti in enumerate(pred.times):
            w.writerow([str(ti),float((ti-pred.epoch)/np.timedelta64(1,'h')),
                        pred.visible_fraction[j],pred.fov_fraction[j],pred.conditional_fov_fraction[j],
                        pred.reference_ra_deg[j],pred.reference_dec_deg[j],pred.reference_elevation_deg[j],
                        bool(pred.visible_fraction[j]>0),bool(pred.reference_elevation_deg[j]>=pred.min_elevation_deg)])
