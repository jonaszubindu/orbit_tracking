"""Monte Carlo dashboard for orbit_mc.MonteCarloResult. Requires scipy."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from orbit_toy import MU, RE, propagate, station_state


def elements(states):
    s=np.atleast_2d(states); r=s[:,:3]; v=s[:,3:]
    R=np.linalg.norm(r,axis=1); h=np.cross(r,v); H=np.linalg.norm(h,axis=1)
    n=np.cross(np.array([0.,0.,1.]),h); N=np.linalg.norm(n,axis=1)
    ev=np.cross(v,h)/MU-r/R[:,None]; e=np.linalg.norm(ev,axis=1)
    a=-MU/(np.sum(v*v,axis=1)-2*MU/R)
    inc=np.degrees(np.arccos(np.clip(h[:,2]/H,-1,1)))
    Om=np.degrees(np.arctan2(n[:,1],n[:,0]))%360
    w=np.degrees(np.arctan2(np.sum(np.cross(n,ev)*h,axis=1)/(N*e*H+1e-30),
                             np.sum(n*ev,axis=1)/(N*e+1e-30)))%360
    f=np.arctan2(np.sum(np.cross(ev,r)*h,axis=1)/(e*R*H+1e-30),
                 np.sum(ev*r,axis=1)/(e*R+1e-30))
    E=2*np.arctan2(np.sqrt(np.maximum(1-e,0))*np.sin(f/2),np.sqrt(1+e)*np.cos(f/2))
    M=np.degrees((E-e*np.sin(E))%(2*np.pi))
    Om[N<1e-8]=np.nan; w[(N<1e-8)|(e<1e-7)]=np.nan; M[e<1e-7]=np.nan
    return np.column_stack((a,e,inc,Om,w,M))


def sky(states, times, epoch, lat, lon, alt):
    dt=(times-epoch)/np.timedelta64(1,'s')
    ras=[]; decs=[]
    for ti,d in zip(times,dt):
        r,_=propagate(states[:,:3].copy(),states[:,3:].copy(),float(d))
        rs,_=station_state(ti,lat,lon,alt); los=r-rs
        ras.append(np.degrees(np.arctan2(los[:,1],los[:,0])))
        decs.append(np.degrees(np.arctan2(los[:,2],np.linalg.norm(los[:,:2],axis=1))))
    return np.array(ras).T,np.array(decs).T


def plot_dashboard(result,times,ra,dec,lat,lon,alt=0.,truth_state=None,truth_epoch=None,
                   output='mc_dashboard.png',max_tracks=350,max_orbit_tracks=18,
                   sky_prediction_minutes=120):
    times=np.asarray(times,dtype='datetime64[ns]'); ra=np.asarray(ra); dec=np.asarray(dec)
    ids=np.flatnonzero(result.accepted)
    if not len(ids): raise ValueError('No threshold-compatible orbits')
    states=result.states[ids]; weights=result.weights[ids].copy()
    if weights.sum()==0: weights[:]=1
    weights/=weights.sum(); el=elements(states)
    # Display angular elements on a continuous, locally unwrapped axis.
    # This prevents a compact cloud near 0/360 degrees from being split.
    for k in (3, 4, 5):
        valid=np.isfinite(el[:, k])
        if not np.any(valid):
            continue
        ang=np.deg2rad(el[valid, k])
        center=np.degrees(np.arctan2(
            np.sum(weights[valid]*np.sin(ang)),
            np.sum(weights[valid]*np.cos(ang)))) % 360
        el[valid, k]=center + (el[valid, k]-center+180)%360-180
    def display_limits(x, w, coverage=(.005, .995)):
        """Weighted central plotting range, robust to extreme tail samples."""
        ok=np.isfinite(x)&np.isfinite(w)&(w>0)
        x=np.asarray(x)[ok]; w=np.asarray(w)[ok]
        if not len(x):
            return (-1., 1.)
        order=np.argsort(x); x=x[order]; w=w[order]
        cdf=(np.cumsum(w)-.5*w)/np.sum(w)
        lo,hi=np.interp(coverage,cdf,x)
        width=hi-lo
        if width<=1e-12*max(abs(lo),abs(hi),1.):
            width=max(abs(lo)*.001,1e-5)
        return (lo-.08*width,hi+.08*width)
    limits=[display_limits(el[:,k],weights) for k in range(6)]
    best=states[np.argmin(result.chi2[ids])]
    # Select a diverse, reproducible subset rather than only the highest-weight states.
    top=np.linspace(0,len(states)-1,min(max_tracks,len(states)),dtype=int)
    ras,decs=sky(states[top],times,result.epoch,lat,lon,alt)
    # Extend the sky predictions to expose the divergence hidden during the fitted arc.
    sky_end=times[-1]+np.timedelta64(int(round(sky_prediction_minutes*60)), 's')
    future_times=np.linspace(times[0].astype('datetime64[s]').astype('int64'),
                             sky_end.astype('datetime64[s]').astype('int64'), 90)
    future_times=np.rint(future_times).astype('int64').astype('datetime64[s]')
    future_ra,future_dec=sky(states[top],future_times,result.epoch,lat,lon,alt)
    best_future_ra,best_future_dec=sky(best[None],future_times,result.epoch,lat,lon,alt)
    bra,bdec=sky(best[None],times,result.epoch,lat,lon,alt)
    wrap=lambda x: ra[0]+((x-ra[0]+180)%360)-180
    ras=wrap(ras); bra=wrap(bra); ra=wrap(ra)
    future_ra=wrap(future_ra);best_future_ra=wrap(best_future_ra)
    t=(times-times[0])/np.timedelta64(1,'s')
    truth_el=None
    if truth_state is not None:
        if truth_epoch is None: truth_epoch=result.epoch
        shift=float((result.epoch-np.datetime64(truth_epoch))/np.timedelta64(1,'s'))
        rt,vt=propagate(np.asarray(truth_state[:3])[None].copy(),np.asarray(truth_state[3:])[None].copy(),shift)
        truth=np.r_[rt[0],vt[0]]; truth_el=elements(truth)[0]
        for k in (3,4,5):
            if np.isfinite(truth_el[k]) and np.any(np.isfinite(el[:,k])):
                center=np.nanmedian(el[:,k])
                truth_el[k]=center+(truth_el[k]-center+180)%360-180
        tra,tdec=sky(truth[None],times,result.epoch,lat,lon,alt); tra=wrap(tra[0]); tdec=tdec[0]
    fig=plt.figure(figsize=(22,14)); outer=fig.add_gridspec(2,3,left=.045,right=.98,top=.93,bottom=.065,
                    width_ratios=[1,1,1],height_ratios=[1,1.15],wspace=.26,hspace=.32)
    fig.suptitle('Angles-only orbit determination | Monte Carlo dashboard',fontsize=19)
    ax=fig.add_subplot(outer[0,0],projection='3d')
    u,v=np.mgrid[0:2*np.pi:26j,0:np.pi:16j]
    ax.plot_surface(RE*np.cos(u)*np.sin(v),RE*np.sin(u)*np.sin(v),RE*np.cos(v),color='steelblue',alpha=.32)
    secs=np.linspace(float((times[0]-result.epoch)/np.timedelta64(1,'s')),5*3600,100)
    for k in np.linspace(0,len(states)-1,min(max_orbit_tracks,len(states)),dtype=int):
        st=states[k]; pts=np.array([propagate(st[:3][None].copy(),st[3:][None].copy(),float(d),max_step=100)[0][0] for d in secs[::4]])
        ax.plot(*pts.T,color='#d77b36',alpha=.55,lw=1.0)
    p=np.array([propagate(best[:3][None].copy(),best[3:][None].copy(),float(d),max_step=100)[0][0] for d in secs])
    ax.plot(*p.T,color='navy',lw=1.8,label='Best sampled orbit')
    obs=np.array([propagate(best[:3][None].copy(),best[3:][None].copy(),float(d))[0][0] for d in (times-result.epoch)/np.timedelta64(1,'s')])
    sc=ax.scatter(*obs.T,c=t,cmap='plasma',s=25,label='Tracklet (inferred 3D)')
    fig.colorbar(sc,ax=ax,shrink=.6,pad=.1,label='Tracklet time [s]')
    ax.set(xlabel='ECI X [km]',ylabel='ECI Y [km]',zlabel='ECI Z [km]',title=f'3D orbit candidates ({min(max_orbit_tracks,len(states))} alternatives)');ax.legend(fontsize=8)
    ax=fig.add_subplot(outer[0,1])
    for a,d in zip(future_ra,future_dec):
        ax.plot(a,d,color='#d77b36',alpha=.24,lw=.85)
    for a,d in zip(ras,decs):
        ax.plot(a,d,color='#458bc4',alpha=.34,lw=1.1)
    ax.plot(best_future_ra[0],best_future_dec[0],color='navy',lw=1.8,
            ls=':',label='Best orbit: extrapolated')
    ax.plot(bra[0],bdec[0],color='navy',lw=2.2,label='Best orbit: observed arc')
    ax.plot([],[],color='#d77b36',lw=1.5,label='Alternative admissible orbits')
    if truth_el is not None: ax.plot(tra,tdec,'r--',label='Synthetic truth')
    ax.scatter(ra,dec,c='black',s=16,label='Measurements')
    s=ax.scatter(ra,dec,c=t,cmap='plasma',s=37,edgecolors='black',linewidths=.4,zorder=4)
    fig.colorbar(s,ax=ax,shrink=.8,label='Time [s]')
    ax.set(xlabel='RA [deg]',ylabel='Dec [deg]',title=f'On-sky tracks (+{sky_prediction_minutes:g} min prediction)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    sub=outer[0,2].subgridspec(4,1,height_ratios=[2,1,2,1],hspace=.2)
    axes=[fig.add_subplot(sub[i]) for i in range(4)]
    for a,d in zip(ras,decs):
        axes[0].plot(t,a,color='#6aa7dd',alpha=.08)
        axes[2].plot(t,d,color='#6aa7dd',alpha=.08)
    for k,obs_y,pred,lab in [(0,ra,bra[0],'RA [deg]'),(2,dec,bdec[0],'Dec [deg]')]:
        axes[k].plot(t,pred,color='navy');axes[k].scatter(t,obs_y,c='black',s=12);axes[k].set_ylabel(lab,fontsize=8)
        axes[k+1].scatter(t,(obs_y-pred)*3600,c='black',s=12)
        axes[k+1].axhline(0,ls='--',color='gray',lw=.8)
        if result.sigma_arcsec is not None: axes[k+1].axhspan(-result.sigma_arcsec,result.sigma_arcsec,color='gray',alpha=.15)
        axes[k+1].set_ylabel('Residual ["]',fontsize=8)
    for a in axes: a.grid(alpha=.15);a.tick_params(labelsize=8)
    axes[-1].set_xlabel('Seconds since first observation');axes[0].set_title('RA/Dec predictions and residuals')
    hgrid=outer[1,0].subgridspec(2,3,wspace=.35,hspace=.36)
    names=['a [km]','e','i [deg]','Ω [deg]','ω [deg]','M [deg]']
    for k in range(6):
        a=fig.add_subplot(hgrid[k//3,k%3]); mask=np.isfinite(el[:,k]); x=el[mask,k]; w=weights[mask]
        if len(x):
            # Use a shared robust window for histogram and correlation panels.
            # Values outside the visible window are deliberately not shown.
            lo,hi=limits[k]
            a.hist(x,bins=25,range=(lo,hi),weights=w/w.sum(),
                   density=True,color='#80b7e6')
            order=np.argsort(x); median=np.interp(.5,np.cumsum(w[order])/w.sum(),x[order])
            a.axvline(median,color='navy',ls='--',label='Weighted median' if k==0 else None)
        if truth_el is not None and np.isfinite(truth_el[k]): a.axvline(truth_el[k],color='red',ls='--',label='Truth' if k==0 else None)
        a.set_title(names[k],fontsize=10);a.tick_params(labelsize=8);a.grid(alpha=.15)
        a.set_xlim(*limits[k])
        if k==0: a.legend(fontsize=7)
    cgrid=outer[1,1].subgridspec(6,6,wspace=.08,hspace=.08)
    for i in range(6):
        for j in range(6):
            a=fig.add_subplot(cgrid[i,j])
            if i<=j: a.axis('off');continue
            x,y=el[:,j],el[:,i]
            xlo,xhi=limits[j]; ylo,yhi=limits[i]
            mask=(np.isfinite(x)&np.isfinite(y)&
                  (x>=xlo)&(x<=xhi)&(y>=ylo)&(y<=yhi))
            if mask.sum()>10 and np.ptp(x[mask])>1e-10 and np.ptp(y[mask])>1e-10:
                H,xe,ye=np.histogram2d(x[mask],y[mask],bins=28,range=[limits[j],limits[i]],weights=weights[mask]);H=gaussian_filter(H,1.1)
                vals=np.sort(H.ravel())[::-1]; cs=np.cumsum(vals)/vals.sum()
                levels=sorted(set([vals[min(np.searchsorted(cs,p),len(vals)-1)] for p in [.9,.5]]+[H.max()*1.001]))
                if len(levels)==3:
                    xc=(xe[1:]+xe[:-1])/2;yc=(ye[1:]+ye[:-1])/2
                    a.contourf(xc,yc,H.T,levels=levels,colors=['#bad9f5','#4d96d2'],alpha=.85)
            if truth_el is not None and np.isfinite(truth_el[i]) and np.isfinite(truth_el[j]):a.plot(truth_el[j],truth_el[i],'r+',ms=5)
            a.set_xlim(*limits[j]);a.set_ylim(*limits[i])
            if i==5:a.set_xlabel(names[j].split(' ')[0],fontsize=8)
            else:a.set_xticklabels([])
            if j==0:a.set_ylabel(names[i].split(' ')[0],fontsize=8)
            else:a.set_yticklabels([])
            a.tick_params(labelsize=6,length=2)
    a=fig.add_subplot(outer[1,2]);a.axis('off')
    ess=1/np.sum(weights**2)
    summary=(f'Monte Carlo summary\n\nDraws: {result.n_drawn:,}\n'
             f'Physically admissible: {len(result.states):,}\n'
             f'Accepted (χ² ≤ {result.max_chi2:g}): {len(ids):,}\n'
             f'ESS, accepted and renormalized: {ess:.1f}\n'
             f'Best sampled χ²: {result.chi2[ids].min():.2f}\n'
             f'Tracklet: {len(times)} points / {t[-1]:.0f} seconds\n'
             f'Measurement σ: {result.sigma_arcsec} arcsec\n\n'
             'Orbital elements: a, e, i, Ω, ω, M\n'
             'at the common fitted reference epoch.\n\n'
             'Pairwise contours: smoothed weighted\n'
             'density; approximate 50% and 90% mass.\n\n'
             'IMPORTANT: These are prior- and\n'
             'threshold-dependent sample summaries,\n'
             'not calibrated posterior intervals.\n'
             'Angular elements wrap at 360 degrees.')
    a.text(.05,.95,summary,va='top',fontsize=11,family='monospace',linespacing=1.5,
           bbox=dict(boxstyle='round,pad=.6',facecolor='#f3f6fa',edgecolor='#d8e0e8'),transform=a.transAxes)
    fig.text(.5,.02,'Toy two-body dynamics; 3D tracklet points are inferred from the orbit, not measured ranges.',ha='center',fontsize=10)
    if output:fig.savefig(output,dpi=145,bbox_inches='tight')
    return fig
