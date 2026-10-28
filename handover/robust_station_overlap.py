"""Fixed-station, scheduled two-minute handover for a known state or local orbit ensemble.

This is a sampled global grid search, NOT a certified continuous-space maximum.
The candidate ensemble is a diagnostic set, NOT a calibrated posterior.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from orbit_toy import RE, station_state, propagate


def unit_vectors(lat,lon):
    a,b=np.deg2rad(np.broadcast_arrays(lat,lon))
    return np.stack((np.cos(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a)),axis=-1)


def load_states(path,mode):
    with np.load(path,allow_pickle=False) as d:
        epoch=np.datetime64(str(d['epoch']),'ns')
        if mode=='known':
            if 'reference_state' in d: states=np.asarray(d['reference_state'],float)[None,:]
            else: states=np.asarray(d['states'],float)[:1]
        else: states=np.asarray(d['states'],float)
    if states.ndim!=(2) or states.shape[1]!=6 or not len(states) or not np.isfinite(states).all():
        raise ValueError('Expected finite states (N,6)')
    return epoch,states,np.ones(len(states))/len(states)


def sat_ecef(states,epoch,t,max_step=60.):
    dt=float((t-epoch)/np.timedelta64(1,'s'))
    r,_=propagate(states[:,:3].copy(),states[:,3:].copy(),dt,max_step=max_step)
    east,_=station_state(t,0.,0.,0.)
    c,s=east[0]/RE,east[1]/RE
    return np.column_stack((c*r[:,0]+s*r[:,1],-s*r[:,0]+c*r[:,1],r[:,2]))


def elevations(sat,station_unit,alt=0.):
    # sat (N,3), station_unit (M,3) -> (N,M) elevation in degrees
    dot=sat@station_unit.T
    ranges=np.sqrt(np.maximum(1e-15,np.sum(sat*sat,axis=1)[:,None]+(RE+alt)**2-2*(RE+alt)*dot))
    return np.rad2deg(np.arcsin(np.clip((dot-(RE+alt))/ranges,-1,1)))


def scan(epoch,states,weights,lat,lon,alt,overlap_min,min_elev,hours,step_min,grid_deg,checks,coverage_levels,station_b_alt=0.):
    if checks<3: raise ValueError('At least 3 overlap checks required')
    if overlap_min<=0 or step_min<=0 or hours<=0 or grid_deg<=0: raise ValueError('Invalid duration/grid')
    lats=np.arange(-90.,90.0001,grid_deg)
    lons=np.arange(-180.,180.,grid_deg)
    glat,glon=np.meshgrid(lats,lons,indexing='ij')
    flatlat,flatlon=glat.ravel(),glon.ravel()
    u=unit_vectors(flatlat,flatlon)
    ua=unit_vectors(np.array([lat]),np.array([lon]))
    sep=RE*np.arccos(np.clip(u@ua[0],-1,1))
    starts=np.arange(0,hours*3600-overlap_min*60+1,step_min*60)
    best_cov=np.zeros(len(u));best_start=np.full(len(u),np.nan)
    rows=[]; best_by_level={float(p):None for p in coverage_levels}
    for sec in starts:
        times=[epoch+np.timedelta64(int(round((sec+tau)*1e9)),'ns') for tau in np.linspace(0,overlap_min*60,checks)]
        viable=np.ones((len(states),len(u)),bool)
        minel_a=np.full(len(states),np.inf)
        minel_b=np.full((len(states),len(u)),np.inf)
        for t in times:
            sat=sat_ecef(states,epoch,t)
            ea=elevations(sat,ua,alt)[:,0]
            eb=elevations(sat,u,station_b_alt)
            minel_a=np.minimum(minel_a,ea)
            minel_b=np.minimum(minel_b,eb)
            viable &= (ea[:,None]>=min_elev)&(eb>=min_elev)
        mass=weights@viable
        better=mass>best_cov+1e-12
        best_cov[better]=mass[better];best_start[better]=sec
        for p in coverage_levels:
            idx=np.flatnonzero(mass+1e-10>=p)
            if not len(idx): continue
            k=int(idx[np.argmax(sep[idx])])
            record={'coverage_threshold':float(p),'distance_km':float(sep[k]),'lat_deg':float(flatlat[k]),'lon_deg':float(flatlon[k]),'start_seconds':float(sec),'ensemble_coverage':float(mass[k]),'minimum_elevation_a_deg':float(np.min(minel_a[viable[:,k]])),'minimum_elevation_b_deg':float(np.min(minel_b[viable[:,k],k]))}
            if best_by_level[float(p)] is None or record['distance_km']>best_by_level[float(p)]['distance_km']:
                best_by_level[float(p)]=record
        rows.append({'start_seconds':float(sec),'max_ensemble_coverage':float(mass.max()),'eligible_locations_90pct':int(np.count_nonzero(mass>=.9))})
    return dict(lats=lats,lons=lons,flatlat=flatlat,flatlon=flatlon,best_cov=best_cov,best_start=best_start,sep=sep,best_by_level=best_by_level,timeline=rows,epoch=epoch,station=(lat,lon,alt),overlap_min=overlap_min,min_elev=min_elev,states=states,weights=weights)


def pointing(states,epoch,t,lat,lon,alt,fov_radius,min_elev):
    sat=sat_ecef(states,epoch,t)
    u=unit_vectors(np.array([lat]),np.array([lon]))[0]
    e=elevations(sat,u[None,:],alt)[:,0]
    v=sat-(RE+alt)*u
    v/=np.linalg.norm(v,axis=1)[:,None]
    good=np.flatnonzero(e>=min_elev)
    if not len(good):return dict(visible_fraction=0.,best_field_fraction=0.,ra_deg=None,dec_deg=None)
    # Reconstruct ECI RA/Dec at the selected time.
    east,_=station_state(t,0.,0.,0.)
    c,s=east[0]/RE,east[1]/RE
    q=np.column_stack((c*v[:,0]-s*v[:,1],s*v[:,0]+c*v[:,1],v[:,2]))
    tree=cKDTree(q[good]); rad=2*np.sin(np.deg2rad(fov_radius)/2)
    neighborhoods=tree.query_ball_point(q[good],rad)
    masses=np.array([len(ids) for ids in neighborhoods],float)/len(states)
    ix=good[np.argmax(masses)]
    return dict(visible_fraction=float(len(good)/len(states)),best_field_fraction=float(masses.max()),ra_deg=float(np.rad2deg(np.arctan2(q[ix,1],q[ix,0]))%360),dec_deg=float(np.rad2deg(np.arcsin(q[ix,2]))))


def great_circle_points(lat, lon, distance_km, n=720):
    """Latitude/longitude of a spherical small circle at constant surface distance."""
    phi, lam = np.deg2rad([lat, lon])
    delta = distance_km / RE
    bearings = np.linspace(0, 2*np.pi, n)
    phis = np.arcsin(np.clip(np.sin(phi)*np.cos(delta) + np.cos(phi)*np.sin(delta)*np.cos(bearings), -1, 1))
    lams = lam + np.arctan2(np.sin(bearings)*np.sin(delta)*np.cos(phi), np.cos(delta)-np.sin(phi)*np.sin(phis))
    return np.rad2deg(phis), (np.rad2deg(lams)+180)%360-180


def plot_handover_world_map(result, pick, mode, path, selected_coverage):
    """World-map coverage with fixed-station great-circle range rings."""
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    try:
        from mpl_toolkits.basemap import Basemap
    except ImportError as exc:
        raise ImportError('World map requires basemap (pip install basemap)') from exc
    fig, ax = plt.subplots(figsize=(14, 7.2), constrained_layout=True)
    ax.set_facecolor('#e8f1f6')
    world = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                    llcrnrlat=-90, urcrnrlat=90, resolution='c', ax=ax)
    world.fillcontinents(color='#e3e7dc', lake_color='#e8f1f6', zorder=1)
    world.drawcoastlines(linewidth=.5, color='#788a91', zorder=2)
    world.drawcountries(linewidth=.3, color='#a0a7a0', zorder=2)
    world.drawparallels(np.arange(-60, 91, 30), labels=[1,0,0,0], linewidth=.35, color='#9aabb5', fontsize=9, dashes=[2,3])
    world.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1], linewidth=.35, color='#9aabb5', fontsize=9, dashes=[2,3])
    grid=result['flatlon']; latgrid=result['flatlat']; coverage=result['best_cov']
    dots=ax.scatter(grid,latgrid,c=coverage,s=39,marker='s',cmap='viridis',
                    vmin=0,vmax=1,alpha=.88,linewidths=0,zorder=3)
    station_lat,station_lon,_=result['station']
    # The rings are geodesic-distance contours, not projected Euclidean circles.
    # Split at the longitude seam to prevent lines crossing the whole map.
    ring_distances=[2000,4000,6000,8000,10000,12000,14000,16000,18000]
    for d in ring_distances:
        ring_lat,ring_lon=great_circle_points(station_lat,station_lon,d)
        jumps=np.flatnonzero(np.abs(np.diff(ring_lon))>180)
        for ii in np.split(np.arange(len(ring_lon)),jumps+1):
            if len(ii)>1:ax.plot(ring_lon[ii],ring_lat[ii],color='#d66d35',alpha=.68,
                                 linewidth=1.05,linestyle='--',zorder=4)
        # Label one point on each ring, preferably avoiding the polar regions.
        target=np.where((np.abs(ring_lat)<72)&(ring_lon>-165)&(ring_lon<165))[0]
        if len(target):
            ix=target[np.argmin(np.abs(ring_lat[target]-5))]
            ax.text(ring_lon[ix],ring_lat[ix],f'{d//1000:,}k',fontsize=7.4,color='#a64b23',
                    ha='center',va='center',zorder=7,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1.1))
    ax.scatter(station_lon,station_lat,s=180,c='#e11d48',marker='*',edgecolor='white',linewidth=1.2,zorder=9)
    ax.annotate('Station A (fixed)',(station_lon,station_lat),xytext=(8,11),textcoords='offset points',
                fontsize=10,fontweight='bold',color='#9f1239',zorder=10,
                bbox=dict(boxstyle='round,pad=.2',facecolor='white',alpha=.9,edgecolor='none'))
    if pick:
        blat,blon=pick['lat_deg'],pick['lon_deg']
        ax.scatter(blon,blat,s=140,c='#22d3ee',marker='D',edgecolor='#083344',linewidth=1.4,zorder=10)
        ax.annotate(f'Station B · {pick["distance_km"]:,.0f} km',(blon,blat),xytext=(9,-18),
                    textcoords='offset points',fontsize=10,fontweight='bold',color='#155e75',zorder=10,
                    bbox=dict(boxstyle='round,pad=.2',facecolor='white',alpha=.92,edgecolor='none'))
        try:world.drawgreatcircle(station_lon,station_lat,blon,blat,linewidth=2.2,color='#0f766e',zorder=8)
        except ValueError:pass
    cbar=fig.colorbar(dots,ax=ax,shrink=.77,pad=.02,aspect=27)
    cbar.set_label('Maximum scheduled ensemble coverage (fraction)',fontsize=10)
    legend=[Line2D([0],[0],marker='s',linestyle='None',color='none',markerfacecolor='#447b7a',
                   markersize=8,label='Candidate station B grid (color = coverage)'),
            Line2D([0],[0],color='#d66d35',ls='--',lw=1.4,label='Surface-distance rings (2,000 km increments)'),
            Line2D([0],[0],marker='*',linestyle='None',color='#e11d48',markersize=13,label='Fixed station A'),
            Line2D([0],[0],marker='D',linestyle='None',color='#22d3ee',markeredgecolor='#083344',markersize=8,label='Best station B'),
            Line2D([0],[0],color='#0f766e',lw=2,label='Shortest great-circle connection')]
    ax.legend(handles=legend,loc='lower left',fontsize=8.5,framealpha=.94)
    ax.set_title(f'Ground-station handover | {mode} orbit | {result["overlap_min"]:g} min overlap',
                 fontsize=15,fontweight='bold',pad=16)
    ax.set_xlabel('Longitude (degrees)');ax.set_ylabel('Latitude (degrees)')
    ax.text(.99,.015,f'Fixed A: {station_lat:.2f}°, {station_lon:.2f}°  |  required coverage: {selected_coverage:.0%}  |  sampled-grid search',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=8.5,color='#334155',
            bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=3))
    fig.savefig(path,dpi=165,bbox_inches='tight');plt.close(fig)


def save(result,out,mode,fov_radius,selected_coverage):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with open(out/'handover_windows.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result['timeline'][0]));w.writeheader();w.writerows(result['timeline'])
    with open(out/'station_grid.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['lat_deg','lon_deg','max_ensemble_coverage','best_start_seconds','separation_km'])
        w.writerows(zip(result['flatlat'],result['flatlon'],result['best_cov'],result['best_start'],result['sep']))
    choices=result['best_by_level'];pick=choices.get(float(selected_coverage))
    if pick:
        t=result['epoch']+np.timedelta64(int(round((pick['start_seconds']+result['overlap_min']*30)*1e9)),'ns')
        pick['receiving_telescope']=pointing(result['states'],result['epoch'],t,pick['lat_deg'],pick['lon_deg'],0.,fov_radius,result['min_elev'])
        pick['sending_telescope']=pointing(result['states'],result['epoch'],t,result['station'][0],result['station'][1],result['station'][2],fov_radius,result['min_elev'])
        pick['elevation_margin_a_deg']=pick['minimum_elevation_a_deg']-result['min_elev']
        pick['elevation_margin_b_deg']=pick['minimum_elevation_b_deg']-result['min_elev']
        pick['midpoint_utc']=str(t)
    summary={'mode':mode,'fixed_station_lat_lon_alt_km':list(result['station']),'overlap_min':result['overlap_min'],'minimum_elevation_deg':result['min_elev'],'candidate_count':len(result['states']),'coverage_interpretation':'exact geometric feasibility for known state; uncalibrated equal candidate fractions for single family','method':'grid and sampled overlap checks; not a certified continuous optimum','best_by_coverage':{str(k):v for k,v in choices.items()},'selected_coverage':selected_coverage,'selected_result':pick}
    (out/'best_handover.json').write_text(json.dumps(summary,indent=2))
    plot_handover_world_map(result, pick, mode, out/'handover_map.png', selected_coverage)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    keys=sorted(choices);ys=[choices[k]['distance_km'] if choices[k] else np.nan for k in keys]
    ax.plot(np.array(keys)*100,ys,'o-',label='Grid maximum separation');ax.set(xlabel='Required ensemble coverage (%)',ylabel='Station separation (km)',title='Separation versus coverage | sampled-grid bound');ax.grid(alpha=.3);ax.legend();fig.savefig(out/'separation_curve.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    ax.plot([x['start_seconds']/3600 for x in result['timeline']],[x['max_ensemble_coverage'] for x in result['timeline']],label='Best station-pair coverage at each start time');ax.set(xlabel='Hours since orbit epoch',ylabel='Ensemble coverage',ylim=(-.05,1.05),title='Two-minute scheduled handover feasibility');ax.grid(alpha=.3);ax.legend();fig.savefig(out/'handover_timeline.png',dpi=160);plt.close(fig)
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--orbit',default='examples/mc_fitted_5_min.npz')
    ap.add_argument('--mode',choices=['known','single'],default='single')
    ap.add_argument('--station-lat',type=float,default=46.95);ap.add_argument('--station-lon',type=float,default=7.44);ap.add_argument('--station-alt-km',type=float,default=.55)
    ap.add_argument('--overlap-min',type=float,default=2);ap.add_argument('--min-elevation',type=float,default=10)
    ap.add_argument('--hours',type=float,default=8);ap.add_argument('--step-min',type=float,default=20);ap.add_argument('--grid-deg',type=float,default=10);ap.add_argument('--checks',type=int,default=5)
    ap.add_argument('--coverage',type=float,default=.9);ap.add_argument('--fov-radius',type=float,default=1)
    ap.add_argument('--output-dir',default='results/robust_handover')
    args=ap.parse_args(argv)
    if not 0<args.coverage<=1:ap.error('--coverage must be in (0,1]')
    if not -90<=args.station_lat<=90 or not -180<=args.station_lon<=180:ap.error('Invalid station coordinates')
    epoch,states,weights=load_states(args.orbit,args.mode)
    levels=sorted(set([.5,.75,.9,.95,1.,args.coverage]))
    res=scan(epoch,states,weights,args.station_lat,args.station_lon,args.station_alt_km,args.overlap_min,args.min_elevation,args.hours,args.step_min,args.grid_deg,args.checks,levels)
    summary=save(res,args.output_dir,args.mode,args.fov_radius,args.coverage)
    print(json.dumps({'selected_result':summary['selected_result'],'candidate_count':len(states),'output_dir':args.output_dir},indent=2))

if __name__=='__main__':main()
