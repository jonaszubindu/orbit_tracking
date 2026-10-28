"""Independent prototype: posterior-weighted or local-ensemble sky-search maps.

Geometric visibility only. Stage B weights are provisional importance weights;
local MC is represented by equal candidate fractions, not calibrated probability.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

# Reuse existing, tested ECI-like station model and vectorized propagator.
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from orbit_station_search import Station, _propagate_ensemble, _angles_and_elevation
from orbit_toy import station_state


def load_orbits(path, mode):
    with np.load(path,allow_pickle=False) as d:
        states=np.asarray(d['states'],float)
        epoch=np.datetime64(str(d['epoch']),'ns')
        if mode=='global':
            weights=np.asarray(d['weights'],float)
            family=np.asarray(d['family_ids'],int) if 'family_ids' in d else np.zeros(len(states),int)
        else:
            weights=np.ones(len(states));family=np.zeros(len(states),int)
    if states.ndim!=2 or states.shape[1]!=6 or not np.isfinite(states).all():
        raise ValueError('Expected finite Cartesian states of shape (N,6)')
    if weights.shape!=(len(states),) or np.any(weights<0) or not np.isfinite(weights).all() or weights.sum()<=0:
        raise ValueError('Invalid sample weights')
    return states,weights/weights.sum(),family,epoch


def sky_geometry(states,epoch,station,hours,step_minutes,min_elev):
    seconds=np.arange(0,hours*3600+1e-6,step_minutes*60.)
    times=epoch+np.rint(seconds*1e9).astype('timedelta64[ns]')
    xyz=_propagate_ensemble(states,seconds,max_step_s=120.)
    st=np.array([station_state(t,station.lat_deg,station.lon_deg,station.altitude_km)[0] for t in times])
    ra,dec,el,unit=_angles_and_elevation(xyz,st)
    return seconds/3600,times,ra,dec,el,unit,el>=min_elev


def optimal_pointing(unit,visible,weights,radius_deg):
    """Find highest-mass circular FoV among sample-centered candidate pointings.

    Discrete candidate centers: not a guaranteed continuous-sky optimum.
    """
    ids=np.flatnonzero(visible)
    if not len(ids):return None,0.,np.zeros(len(weights),bool)
    uv=unit[ids]
    cosrad=np.cos(np.deg2rad(radius_deg))
    best=-1.;bestid=ids[0];bestmask=None
    # Chunking avoids N*N storage for large global ensembles.
    for start in range(0,len(ids),96):
        center=uv[start:start+96]
        in_field=(center@uv.T)>=cosrad
        scores=in_field@weights[ids]
        j=int(np.argmax(scores))
        if scores[j]>best:
            best=float(scores[j]);bestid=int(ids[start+j]);bestmask=in_field[j]
    mask=np.zeros(len(weights),bool);mask[ids]=bestmask
    return bestid,best,mask


def tangent_offsets(unit,center):
    z=np.array([0.,0.,1.])
    east=np.cross(z,center)
    if np.linalg.norm(east)<1e-8:east=np.cross([0.,1.,0.],center)
    east=east/np.linalg.norm(east)
    north=np.cross(center,east)
    dot=unit@center
    # Gnomonic projection in degrees, valid for the selected local sky patch.
    good=dot>0
    x=np.full(len(unit),np.nan);y=x.copy()
    x[good]=np.rad2deg((unit[good]@east)/dot[good])
    y[good]=np.rad2deg((unit[good]@north)/dot[good])
    return x,y,good


def generate(path,mode,station,out,hours=12,step_minutes=20,min_elev=10,fov_radius=1.,grid_radius=12.,bins=85):
    states,weights,families,epoch=load_orbits(path,mode)
    hours_axis,times,ra,dec,el,unit,visible=sky_geometry(states,epoch,station,hours,step_minutes,min_elev)
    visprob=visible@weights
    bestprob=np.zeros(len(times));bestidx=np.full(len(times),-1,int)
    for j in range(len(times)):
        ix,p,_=optimal_pointing(unit[j],visible[j],weights,fov_radius)
        bestprob[j]=p
        if ix is not None:bestidx[j]=ix
    if not np.any(visible):
        selected=int(np.argmax(visprob));flag='NO PASS'
    else:
        # Show the time with the largest probability in a single pointing.
        selected=int(np.argmax(bestprob));flag='PASS POSSIBLE'
    center=unit[selected,bestidx[selected]] if bestidx[selected]>=0 else unit[selected,np.argmax(weights)]
    x,y,front=tangent_offsets(unit[selected],center)
    keep=visible[selected]&front&np.isfinite(x)&np.isfinite(y)
    lim=grid_radius
    inside=keep&(np.abs(x)<=lim)&(np.abs(y)<=lim)
    H,xe,ye=np.histogram2d(x[inside],y[inside],bins=bins,range=[[-lim,lim],[-lim,lim]],weights=weights[inside])
    # Display smoothing only; raw probabilities computed from unsmoothed particles.
    smooth=gaussian_filter(H,1.1)
    fig=plt.figure(figsize=(13.8,9.2),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.45])
    ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1]);cx=fig.add_subplot(gs[1,:])
    ax.plot(hours_axis,visprob,label='Above elevation threshold',lw=2,color='#1f77b4')
    ax.plot(hours_axis,bestprob,label=f'Best {fov_radius:g}°-radius field',lw=2,color='#d27d18')
    ax.axvline(hours_axis[selected],ls=':',color='black',alpha=.7,label='Time shown in sky map')
    ax.set(xlabel='Hours since state epoch',ylabel='Ensemble mass / candidate fraction',ylim=(-.025,1.025))
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fams=np.unique(families)
    for fid in fams:
        fm=(families==fid)
        if fm.sum()==0:continue
        bx.plot(hours_axis,(visible[:,fm]@weights[fm]),label=f'Family {fid} (mass {weights[fm].sum():.1%})',lw=1.7)
    bx.axvline(hours_axis[selected],ls=':',color='black',alpha=.7,label='Time shown in sky map')
    bx.set(xlabel='Hours since state epoch',ylabel='Visible mass per family',ylim=(-.025,1.025))
    bx.grid(alpha=.2);bx.legend(fontsize=8,ncol=2)
    mesh=cx.pcolormesh(xe,ye,smooth.T,cmap='magma',shading='auto')
    fig.colorbar(mesh,ax=cx,label='Smoothed weighted mass per map cell (display only)',shrink=.85)
    cx.scatter(x[inside],y[inside],s=3,color='#7cd7f1',alpha=.22,rasterized=True,label='Individual visible orbit samples')
    cx.add_patch(Circle((0,0),fov_radius,fill=False,color='cyan',lw=2,label='Best sample-centered FoV'))
    cx.scatter([0],[0],marker='+',color='cyan',s=130,linewidths=2,label='Optimal pointing center (0°, 0°)')
    cx.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal',xlabel='Local east tangent offset [deg]',ylabel='Local north tangent offset [deg]',title=f'Search map at {str(times[selected])[:16]} UTC  |  best field {bestprob[selected]:.1%}  |  visible {visprob[selected]:.1%}')
    cx.grid(alpha=.15)
    cx.legend(handles=[
        Patch(facecolor='#a64a82',edgecolor='none',label='Heatmap: smoothed weighted orbit density'),
        Line2D([],[],marker='.',linestyle='None',color='#7cd7f1',markersize=9,label='Cyan dots: individual visible orbit samples'),
        Line2D([],[],marker='o',linestyle='None',markerfacecolor='none',markeredgecolor='cyan',markersize=12,label=f'Cyan circle: {fov_radius:g}°-radius telescope field'),
        Line2D([],[],marker='+',linestyle='None',color='cyan',markersize=12,markeredgewidth=2,label='Cyan +: chosen pointing center'),
    ],loc='upper right',fontsize=8,framealpha=.88)
    kind='provisional Stage B weighted mass' if mode=='global' else 'equal-weight local candidate fraction'
    fig.suptitle(f'{station.name}  |  {flag}  |  {mode.upper()} mode\n{kind}; geometric horizon only, not optical detection probability',fontsize=13)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=155);plt.close(fig)
    csv=out.with_suffix('.csv')
    np.savetxt(csv,np.column_stack([hours_axis,visprob,bestprob]),delimiter=',',header='hours_since_epoch,visible_mass,best_fov_mass',comments='')
    print(f'{mode} {station.name}: {flag}; peak visible {visprob.max():.1%}; best field {bestprob.max():.1%}; saved {out}')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['single','global'],required=True)
    p.add_argument('--input',required=True,help='Existing MC NPZ or Stage B NPZ')
    p.add_argument('--lat',type=float,required=True);p.add_argument('--lon',type=float,required=True)
    p.add_argument('--alt-km',type=float,default=0.)
    p.add_argument('--name',default='Custom station')
    p.add_argument('--hours',type=float,default=12)
    p.add_argument('--step-minutes',type=float,default=20)
    p.add_argument('--min-elevation',type=float,default=10)
    p.add_argument('--fov-radius',type=float,default=1)
    p.add_argument('--map-radius',type=float,default=12)
    p.add_argument('--output',default='search_map.png')
    a=p.parse_args()
    generate(a.input,a.mode,Station(a.name,a.lat,a.lon,a.alt_km),a.output,a.hours,a.step_minutes,a.min_elevation,a.fov_radius,a.map_radius)
if __name__=='__main__':main()
