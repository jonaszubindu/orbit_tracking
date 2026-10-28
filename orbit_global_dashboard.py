"""Independent global-family dashboard; does not modify the legacy dashboard.

Stage B weights are importance weights, not calibrated posterior probabilities.
Family labels are nearest Stage A seeds, not verified connected components.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from orbit_toy import propagate, station_state, RE


@dataclass
class GlobalEnsemble:
    epoch: np.datetime64
    station: tuple
    states: np.ndarray
    weights: np.ndarray
    family_ids: np.ndarray
    chi2: np.ndarray
    ess: float
    times: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    stage_a_states: np.ndarray
    stage_a_families: np.ndarray
    reference_state: np.ndarray | None = None
    local_states: np.ndarray | None = None


def load_ensemble(stage_a_path, stage_b_path, local_mc_path=None):
    with np.load(stage_a_path) as a, np.load(stage_b_path) as b:
        epoch=np.datetime64(b['epoch'],'ns')
        if epoch != np.datetime64(a['epoch'],'ns'):
            raise ValueError('Stage A and B state epochs differ')
        states=np.asarray(b['states'],float)
        weights=np.asarray(b['weights'],float)
        if states.ndim!=2 or states.shape[1]!=6 or weights.shape!=(len(states),):
            raise ValueError('Invalid Stage B state/weight dimensions')
        if not np.all(np.isfinite(weights)) or np.any(weights<0) or weights.sum()<=0:
            raise ValueError('Invalid Stage B weights')
        weights=weights/weights.sum()
        station=tuple(np.asarray(b['station'],float))
        a_station=(float(a['station_lat_deg']),float(a['station_lon_deg']),float(a['station_alt_km']))
        if not np.allclose(station,a_station):raise ValueError('Stage A/B stations differ')
        result=GlobalEnsemble(epoch,station,states,weights,np.asarray(b['family_ids'],int),
            np.asarray(b['chi2'],float),float(b['ess']),np.asarray(a['times'],dtype='datetime64[ns]'),
            np.asarray(a['ra_deg'],float),np.asarray(a['dec_deg'],float),
            np.asarray(a['states'],float),np.asarray(a['family_ids'],int))
    if local_mc_path is not None:
        with np.load(local_mc_path) as m:
            lepoch=np.datetime64(str(m['epoch']),'ns')
            shift=float((epoch-lepoch)/np.timedelta64(1,'s'))
            def move(s):
                r,v=propagate(s[:,:3].copy(),s[:,3:].copy(),shift,max_step=90.)
                return np.column_stack((r,v))
            result.local_states=move(np.asarray(m['states'],float))
            result.reference_state=move(np.asarray(m['reference_state'],float)[None])[0]
    return result


def weighted_quantile(values, weights, q):
    ix=np.argsort(values);v=np.asarray(values)[ix];w=np.asarray(weights)[ix]
    if not len(v) or w.sum()<=0:return np.full(len(np.atleast_1d(q)),np.nan)
    c=(np.cumsum(w)-.5*w)/w.sum()
    return np.interp(q,c,v)


def stratified_indices(ids,weights,n,rng):
    """Probability-proportional systematic resampling, for DISPLAY only."""
    if len(ids)==0 or n<=0:return np.array([],int)
    w=weights[ids];w=w/w.sum()
    c=np.cumsum(w)
    u=(rng.random()+np.arange(n))/n
    return ids[np.searchsorted(c,u,side='left').clip(max=len(ids)-1)]


def sky(states,times,epoch,station,max_step=90.):
    ra=[];dec=[];rr=[]
    for t in times:
        dt=float((t-epoch)/np.timedelta64(1,'s'))
        r,_=propagate(states[:,:3].copy(),states[:,3:].copy(),dt,max_step=max_step)
        st,_=station_state(t,*station)
        d=r-st
        ra.append(np.degrees(np.arctan2(d[:,1],d[:,0])))
        dec.append(np.degrees(np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))))
        rr.append(r)
    return np.array(ra).T,np.array(dec).T,np.array(rr).transpose(1,0,2)


def orbital_elements(states):
    from orbit_dashboard_elements import elements
    return elements(states)


def plot_global_dashboard(data,output,prediction_minutes=120,max_tracks_per_family=16,
                          max_element_samples=3000,seed=43):
    rng=np.random.default_rng(seed)
    w=data.weights; fam=data.family_ids
    active=np.flatnonzero(w>0)
    families=sorted(int(f) for f in np.unique(fam[active]))
    if not families:raise ValueError('No positive-weight Stage B states')
    cmap=plt.get_cmap('tab10')
    colors={f:cmap(j%10) for j,f in enumerate(families)}
    mass={f:float(w[fam==f].sum()) for f in families}
    family_ess={f:float(mass[f]**2/np.sum(w[fam==f]**2)) for f in families}
    chosen={f:stratified_indices(np.flatnonzero((fam==f)&(w>0)),w,max_tracks_per_family,rng) for f in families}
    times=np.linspace(0,prediction_minutes,65)
    prediction=data.times[-1]+np.rint(times*60).astype('timedelta64[s]')
    observed_ra=data.ra_deg[0]
    wrap=lambda x:observed_ra+((x-observed_ra+180)%360)-180
    fig=plt.figure(figsize=(21,14),layout='constrained')
    gs=fig.add_gridspec(2,3,height_ratios=[1,1.1],width_ratios=[1.1,1.05,1.1])
    ax3=fig.add_subplot(gs[0,0],projection='3d')
    axsky=fig.add_subplot(gs[0,1]);axres=fig.add_subplot(gs[0,2]);axsum=fig.add_subplot(gs[1,0]);
    sub=gs[1,1:].subgridspec(2,3,wspace=.22,hspace=.38)
    axs=[fig.add_subplot(sub[i//3,i%3]) for i in range(6)]
    phi,theta=np.mgrid[0:2*np.pi:18j,0:np.pi:12j]
    ax3.plot_surface(RE*np.cos(phi)*np.sin(theta),RE*np.sin(phi)*np.sin(theta),RE*np.cos(theta),color='steelblue',alpha=.25)
    all_points=[]
    for f in families:
        idx=chosen[f]
        if len(idx)==0:continue
        ras,decs,xyz=sky(data.states[idx],prediction,data.epoch,data.station)
        # Normalize RA relative to first observation; family tracks share the same frame.
        for k in range(len(idx)):
            axsky.plot(wrap(ras[k]),decs[k],color=colors[f],alpha=.23,lw=.9)
            ax3.plot(*xyz[k,::3,:].T,color=colors[f],alpha=.23,lw=.8)
        all_points.append(xyz)
        # Highest-weight state is a visual representative, not a posterior mode estimate.
        top=np.flatnonzero((fam==f)&(w>0))[np.argmax(w[(fam==f)&(w>0)])]
        rr,dd,xx=sky(data.states[top:top+1],prediction,data.epoch,data.station)
        axsky.plot(wrap(rr[0]),dd[0],color=colors[f],lw=2.2,label=f'Family {f} ({mass[f]:.1%})')
        ax3.plot(*xx[0,::2,:].T,color=colors[f],lw=2.1)
        obs_r,obs_d,_=sky(data.states[top:top+1],data.times,data.epoch,data.station)
        residual_ra=(wrap(obs_r[0])-wrap(data.ra_deg))*3600
        residual_dec=(obs_d[0]-data.dec_deg)*3600
        tt=(data.times-data.times[0])/np.timedelta64(1,'s')/60
        axres.plot(tt,residual_ra,color=colors[f],lw=1.2,alpha=.8)
        axres.plot(tt,residual_dec,color=colors[f],ls='--',lw=1.2,alpha=.8)
    if data.reference_state is not None:
        rr,dd,xx=sky(data.reference_state[None],prediction,data.epoch,data.station)
        axsky.plot(wrap(rr[0]),dd[0],color='black',ls='--',lw=1.8,label='Legacy fit (diagnostic)')
        ax3.plot(*xx[0,::2,:].T,color='black',ls='--',lw=1.8)
    if data.local_states is not None and len(data.local_states):
        ix=rng.choice(len(data.local_states),size=min(8,len(data.local_states)),replace=False)
        rr,dd,xx=sky(data.local_states[ix],prediction,data.epoch,data.station)
        for k in range(len(ix)):
            axsky.plot(wrap(rr[k]),dd[k],color='gray',ls=':',lw=.9,alpha=.5)
            ax3.plot(*xx[k,::3,:].T,color='gray',ls=':',lw=.8,alpha=.4)
    axsky.scatter(wrap(data.ra_deg),data.dec_deg,c='black',s=11,zorder=8,label='Observations')
    axsky.set(xlabel='RA (unwrapped, deg)',ylabel='Dec (deg)',title='On-sky tracks + future extrapolation')
    axsky.legend(fontsize=8,loc='best')
    axres.axhline(0,color='gray',lw=.6)
    axres.set(xlabel='Time since first observation (min)',ylabel='Residual (arcsec)',title='Family representative RA (solid), Dec (dashed)')
    ax3.set(xlabel='ECI x (km)',ylabel='ECI y (km)',zlabel='ECI z (km)',title='Orbit families, future segment')
    if all_points:
        pts=np.concatenate(all_points,axis=0).reshape(-1,3)
        lim=max(RE*1.3,float(np.nanpercentile(np.abs(pts),97)))
        for setter in (ax3.set_xlim,ax3.set_ylim,ax3.set_zlim):setter(-lim,lim)
    axsum.set_axis_off()
    rows=[f'Family {f}: weight {mass[f]:.2%} | within-family ESS {family_ess[f]:.1f}' for f in families]
    summary='GLOBAL FAMILY ENSEMBLE\n\n'+ '\n'.join(rows)
    summary+=f'\n\nTotal ESS: {data.ess:.1f} / {len(w)} draws'
    summary+='\n\nLabels = nearest Stage A seed, NOT proven modes.'
    summary+='\nWeights = provisional, prior-dependent importance weights.'
    summary+='\nTransparent curves = within-family weighted resamples.'
    summary+='\nGray dotted = legacy local MC (unweighted diagnostic).'
    summary+='\nBlack dashed = legacy deterministic fit.'
    axsum.text(.02,.98,summary,transform=axsum.transAxes,va='top',ha='left',fontsize=10,
               family='monospace',linespacing=1.55)
    # Sample display points proportionally to posterior mass while retaining family colors.
    sample_ids=stratified_indices(active,w,min(max_element_samples,len(active)),rng)
    els=orbital_elements(data.states[sample_ids]);names=['a (km)','e','i (deg)','RAAN (deg)','arg periapsis (deg)','M (deg)']
    for j,ax in enumerate(axs):
        values=els[:,j].copy()
        ok=np.isfinite(values)
        if j>=3 and np.any(ok):
            phase=np.degrees(np.angle(np.mean(np.exp(1j*np.deg2rad(values[ok])))))
            values[ok]=phase+(values[ok]-phase+180)%360-180
        if not np.any(ok):continue
        lo,hi=np.quantile(values[ok],[.005,.995]);span=max(hi-lo,1e-8)
        bins=np.linspace(lo-.04*span,hi+.04*span,35)
        for f in families:
            mask=(fam[sample_ids]==f)&ok
            if mask.any():ax.hist(values[mask],bins=bins,histtype='step',lw=1.8,color=colors[f],label=f'Family {f}')
        ax.set_title(names[j],fontsize=10);ax.set_ylabel('Resampled count',fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle('Global orbit families | separate Stage A+B dashboard\n'
                 'Importance-weighted display, provisional family labels; not calibrated probabilities',fontsize=16)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=130);plt.close(fig)
    return {'family_mass':mass,'family_ess':family_ess,'ess':data.ess,'output':str(output)}
