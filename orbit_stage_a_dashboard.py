"""Stage A global family dashboard; families are discovered candidates, not probabilities."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from orbit_fit_multipass import propagate_at,elements
from orbit_toy import station_state,RE


def plot_stage_a_dashboard(result,output,truth_state=None,truth_epoch=None,max_tracks=45,prediction_minutes=120):
    fig=plt.figure(figsize=(16,10),layout='constrained')
    gs=fig.add_gridspec(2,3)
    axmap=fig.add_subplot(gs[0,0]);axsky=fig.add_subplot(gs[0,1]);ax3d=fig.add_subplot(gs[0,2],projection='3d')
    axchi=fig.add_subplot(gs[1,0]);axa=fig.add_subplot(gs[1,1]);axe=fig.add_subplot(gs[1,2])
    xx,yy=np.meshgrid(result.ranges_km,result.rates_km_s,indexing='ij')
    good=np.isfinite(result.grid_chi2)
    values=np.ma.masked_where(~good,np.maximum(result.grid_chi2-result.best_chi2,0)+1)
    axmap.pcolormesh(xx,yy,values,norm=LogNorm(vmin=1,vmax=max(10,float(np.nanpercentile(values.compressed(),85)))),cmap='viridis',shading='auto')
    axmap.set_xscale('log');axmap.set_xlabel('Range ρ [km]');axmap.set_ylabel('Range rate ρ̇ [km/s]')
    axmap.set_title('Coarse fixed-attributable χ² map')
    st=station_state(result.epoch,*result.station)[0]
    rho=np.linalg.norm(result.states[:,:3]-st,axis=1)
    u=(result.states[:,:3]-st)/rho[:,None]
    vst=station_state(result.epoch,*result.station)[1]
    rate=np.sum((result.states[:,3:]-vst)*u,axis=1)
    colors=plt.get_cmap('tab10')
    for fid in np.unique(result.family_ids):
        sel=result.family_ids==fid;c=colors((fid-1)%10)
        axmap.scatter(rho[sel],rate[sel],s=30,c=[c],edgecolors='white',linewidths=.4,label=f'Family {fid}')
    axmap.legend(fontsize=8,loc='best')
    tstart=float((result.times[0]-result.epoch)/np.timedelta64(1,'s'))
    tend=float((result.times[-1]-result.epoch)/np.timedelta64(1,'s'))+60*prediction_minutes
    tt=np.linspace(tstart,tend,100)
    ts=result.epoch+np.rint(tt*1e9).astype('timedelta64[ns]')
    station=np.array([station_state(t,*result.station)[0] for t in ts])
    # Use all found candidates up to display cap, stratified by family.
    chosen=[]
    for fid in np.unique(result.family_ids):
        inds=np.flatnonzero(result.family_ids==fid)
        chosen.extend(inds[np.linspace(0,len(inds)-1,min(len(inds),max(1,max_tracks//result.family_count)),dtype=int)])
    chosen=chosen[:max_tracks]
    orbit_t=np.linspace(tstart,min(tend,14400),90)
    for k in chosen:
        state=result.states[k];fid=result.family_ids[k];c=colors((fid-1)%10)
        p=propagate_at(state,tt)[:,:3]-station
        ra=np.rad2deg(np.unwrap(np.arctan2(p[:,1],p[:,0])))
        dec=np.rad2deg(np.arctan2(p[:,2],np.hypot(p[:,0],p[:,1])))
        axsky.plot(ra,dec,color=c,alpha=.48,lw=1)
        orbit=propagate_at(state,orbit_t)[:,:3]
        ax3d.plot(*orbit.T,color=c,alpha=.4,lw=.8)
    obsra=np.rad2deg(np.unwrap(np.deg2rad(result.ra_deg)))
    axsky.scatter(obsra,result.dec_deg,c='black',s=10,label='Observations',zorder=5)
    axsky.set_xlabel('Unwrapped RA [deg]');axsky.set_ylabel('Dec [deg]');axsky.set_title('Observed and extrapolated sky tracks')
    axsky.legend(fontsize=8)
    # Earth at origin
    a,b=np.meshgrid(np.linspace(0,2*np.pi,20),np.linspace(0,np.pi,10))
    ax3d.plot_wireframe(RE*np.cos(a)*np.sin(b),RE*np.sin(a)*np.sin(b),RE*np.cos(b),alpha=.13,color='grey')
    ax3d.set_xlabel('ECI x [km]');ax3d.set_ylabel('ECI y [km]');ax3d.set_zlabel('ECI z [km]')
    ax3d.set_title('Candidate orbit segments (colored by family)')
    for fid in np.unique(result.family_ids):
        mask=result.family_ids==fid;c=colors((fid-1)%10)
        axchi.scatter(rho[mask],result.chi2[mask]-result.best_chi2,color=c,s=20,label=f'Family {fid}')
        els=[elements(s) for s in result.states[mask]]
        axa.scatter([e['a_km'] for e in els],[e['e'] for e in els],s=22,color=c)
        axe.scatter([e['i_deg'] for e in els],[e['raan_deg'] for e in els],s=22,color=c)
    axchi.axhline(result.threshold-result.best_chi2,color='grey',ls='--',label='Selection cutoff')
    axchi.set_xscale('log');axchi.set_xlabel('Range [km]');axchi.set_ylabel('Δχ²');axchi.set_title('Refined candidate fits');axchi.legend(fontsize=8)
    axa.set_xlabel('Semimajor axis [km]');axa.set_ylabel('Eccentricity');axa.set_title('Discovered orbital-element families')
    axe.set_xlabel('Inclination [deg]');axe.set_ylabel('RAAN [deg]');axe.set_title('Orbital plane families')
    for ax in (axmap,axsky,axchi,axa,axe):ax.grid(alpha=.15)
    fig.suptitle(f'Stage A | {len(result.states)} refined compatible candidates | {result.family_count} provisional families | NOT probabilities',fontsize=14)
    fig.savefig(output,dpi=140);plt.close(fig)
