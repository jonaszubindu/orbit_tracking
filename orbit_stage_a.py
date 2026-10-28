"""Stage A: global admissible-region discovery, NOT posterior inference.

Search rho/rho_dot on a bounded grid. Optimize the 4 angular attributable
coordinates independently at selected grid seeds, then cluster accepted
solutions in (log rho, rho_dot) into connected families.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label, maximum_filter
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2dist
from tracklet import fit_tracklet
from orbit_toy import ARCSEC, orbital_filter, station_state, initial_state, propagate
from orbit_fit_multipass import propagate_at

@dataclass
class StageAResult:
    epoch: np.datetime64
    ranges_km: np.ndarray
    rates_km_s: np.ndarray
    grid_chi2: np.ndarray
    grid_physical: np.ndarray
    grid_family: np.ndarray
    states: np.ndarray
    chi2: np.ndarray
    family_ids: np.ndarray
    family_count: int
    threshold: float
    best_chi2: float
    attribution: object
    times: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    station: tuple


def _state(a, rho, rate, epoch, station):
    from types import SimpleNamespace
    attr=SimpleNamespace(ra_deg=np.rad2deg(a[0]),dec_deg=np.rad2deg(a[1]),
       ra_rate_deg_s=np.rad2deg(a[2]),dec_rate_deg_s=np.rad2deg(a[3]))
    r,v=initial_state(attr,station,np.array([rho]),np.array([rate]))
    return np.r_[r[0],v[0]]


def search_stage_a(times, ra_deg, dec_deg, lat_deg, lon_deg, altitude_km=0.,
                   sigma_arcsec=5., range_bounds=(300.,60000.),
                   rate_bounds=(-10.,10.), n_range=45, n_rate=51,
                   min_perigee_alt_km=100., delta_chi2=16.812,
                   n_refine=40, max_nfev=35, max_step_s=20.):
    """Explore all cells of a bounded grid; refinement is heuristic, not exhaustive.

    delta_chi2=16.812 corresponds to a six-parameter 99% likelihood-ratio
    contour under regular asymptotic assumptions, not a confidence guarantee.
    Grid chi2 holds *fixed-attributable* scores; refined states allow all four
    angular parameters to vary. Grid family labels are connected components
    of the coarse approximate contour, not guaranteed global orbit families.
    """
    times=np.asarray(times,dtype='datetime64[ns]'); ra=np.asarray(ra_deg,float);dec=np.asarray(dec_deg,float)
    if len(times)<3 or np.any(np.diff(times)<=np.timedelta64(0,'ns')): raise ValueError('Need increasing times')
    attr=fit_tracklet(times,ra,dec,sigma_arcsec)
    epoch=attr.epoch; station=station_state(epoch,lat_deg,lon_deg,altitude_km)
    a0=np.deg2rad([attr.ra_deg,attr.dec_deg,attr.ra_rate_deg_s,attr.dec_rate_deg_s]);
    ranges=np.geomspace(*range_bounds,n_range); rates=np.linspace(*rate_bounds,n_rate)
    rr,rv=np.meshgrid(ranges,rates,indexing='ij')
    r0,v0=initial_state(attr,station,rr.ravel(),rv.ravel())
    physical,_=orbital_filter(r0,v0,min_perigee_alt_km)
    scores=np.full(len(rr.ravel()),np.inf)
    ids=np.flatnonzero(physical)
    if not len(ids): raise ValueError('No physical grid states')
    scores[ids]=0
    # Integrate all physical candidates in parallel, one propagation per observation.
    for ti,oa,od in zip(times,np.deg2rad(ra),np.deg2rad(dec)):
        dt=float((ti-epoch)/np.timedelta64(1,'s'))
        r,_=propagate(r0[ids].copy(),v0[ids].copy(),dt,max_step=max_step_s)
        st=station_state(ti,lat_deg,lon_deg,altitude_km)[0]
        d=r-st; pa=np.arctan2(d[:,1],d[:,0]);pd=np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))
        da=np.arctan2(np.sin(pa-oa),np.cos(pa-oa))
        scores[ids]+=((da*ARCSEC/sigma_arcsec)**2+((pd-od)*ARCSEC/sigma_arcsec)**2)
    grid=scores.reshape(rr.shape);phys=physical.reshape(rr.shape)
    # Include distinct local minima, not merely the globally lowest cells.
    minima=(grid== -maximum_filter(-grid,size=3,mode='nearest')) & phys
    seeds=list(zip(*np.where(minima)))
    seeds.sort(key=lambda ij:grid[ij])
    extra=list(zip(*np.unravel_index(np.argsort(scores)[:max(n_refine*4,40)],grid.shape)))
    for ij in extra:
        if tuple(ij) not in seeds:seeds.append(tuple(ij))
    seeds=seeds[:n_refine]
    offsets=np.array([(ti-epoch)/np.timedelta64(1,'s') for ti in times],float)
    stations=np.array([station_state(ti,lat_deg,lon_deg,altitude_km)[0] for ti in times])
    oa=np.deg2rad(ra);od=np.deg2rad(dec)
    # Angular parameters are scaled by their attributable 1-sigma uncertainties.
    scales=np.sqrt(np.maximum(np.diag(attr.covariance),1e-24))
    def residual(z,rho,rate):
        state=_state(a0+z*scales,rho,rate,epoch,station)
        try: d=propagate_at(state,offsets,rtol=1e-9,atol=1e-9)[:,:3]-stations
        except (ValueError,RuntimeError):return np.full(2*len(times),1e7)
        pa=np.arctan2(d[:,1],d[:,0]);pd=np.arctan2(d[:,2],np.hypot(d[:,0],d[:,1]))
        da=np.arctan2(np.sin(pa-oa),np.cos(pa-oa))
        return np.column_stack((da,pd-od)).ravel()*ARCSEC/sigma_arcsec
    refined=[]
    for i,j in seeds:
        rho,rate=float(ranges[i]),float(rates[j]); opt=least_squares(residual,np.zeros(4),args=(rho,rate),max_nfev=max_nfev,
            method='trf',ftol=1e-7,xtol=1e-7,gtol=1e-7)
        state=_state(a0+opt.x*scales,rho,rate,epoch,station)
        ok,_=orbital_filter(state[:3][None],state[3:][None],min_perigee_alt_km)
        if ok[0]:refined.append((float(opt.fun@opt.fun),rho,rate,state))
    if not refined:raise RuntimeError('No physically admissible refined candidates')
    refined.sort(key=lambda x:x[0]);best=refined[0][0];limit=best+delta_chi2
    # Coarse connected components: relaxed to include fixed-attributable error,
    # which can be very large when the tracklet has measurable curvature.
    # This is a visualization heuristic, NOT a posterior credible region.
    coarse_limit=max(np.min(grid[np.isfinite(grid)])+max(delta_chi2,20.),limit)
    mask=phys & (grid<=coarse_limit)
    components,ncomp=label(mask,np.ones((3,3),int))
    accepted=[item for item in refined if item[0]<=limit]
    states=np.array([v[3] for v in accepted]);chis=np.array([v[0] for v in accepted])
    family=np.zeros(len(accepted),int)
    # Assign refined points to nearest connected component, or create a
    # singleton when no coarse component is nearby.
    nextid=ncomp
    for k,(_,rho,rate,_) in enumerate(accepted):
        i=np.argmin(abs(np.log(ranges/rho)));j=np.argmin(abs(rates-rate))
        lab=int(components[i,j]);
        if lab==0:
            nextid+=1;lab=nextid
        family[k]=lab
    unique=sorted(set(family));remap={v:i+1 for i,v in enumerate(unique)}
    family=np.array([remap[v] for v in family]);
    # Grid labels do not necessarily match refined family IDs; keep separately.
    return StageAResult(epoch,ranges,rates,grid,phys,components,states,chis,family,
        len(unique),limit,best,attr,times,ra,dec,(lat_deg,lon_deg,altitude_km))
