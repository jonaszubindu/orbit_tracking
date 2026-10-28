"""Deterministic angles-only orbit fitting: circular seeds -> free 6D least squares.

Two-body Earth dynamics, matching orbit_toy.station_state. No MCMC.
Times may span multiple observing passes; they need not be uniformly spaced.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from orbit_toy import MU, RE, ARCSEC, station_state, orbital_filter


def elements(state):
    r,v=np.asarray(state[:3]),np.asarray(state[3:]); rn=np.linalg.norm(r)
    h=np.cross(r,v); hn=np.linalg.norm(h); n=np.cross([0.,0.,1.],h); nn=np.linalg.norm(n)
    ev=np.cross(v,h)/MU-r/rn; e=np.linalg.norm(ev); a=1/(2/rn-np.dot(v,v)/MU)
    inc=np.arccos(np.clip(h[2]/hn,-1,1))
    raan=np.arctan2(n[1],n[0])%(2*np.pi) if nn>1e-10 else np.nan
    argp=np.arctan2(np.dot(np.cross(n,ev),h)/max(nn*e*hn,1e-20),np.dot(n,ev)/max(nn*e,1e-20))%(2*np.pi) if nn>1e-10 and e>1e-9 else np.nan
    nu=np.arctan2(np.dot(np.cross(ev,r),h)/max(e*rn*hn,1e-20),np.dot(ev,r)/max(e*rn,1e-20)) if e>1e-9 else np.nan
    if e<1 and np.isfinite(nu):
        E=2*np.arctan2(np.sqrt(1-e)*np.sin(nu/2),np.sqrt(1+e)*np.cos(nu/2)); M=(E-e*np.sin(E))%(2*np.pi)
    else:M=np.nan
    return dict(a_km=a,e=e,i_deg=np.rad2deg(inc),raan_deg=np.rad2deg(raan),argp_deg=np.rad2deg(argp),M_deg=np.rad2deg(M))


def propagate_at(state, seconds, rtol=2e-11, atol=1e-11):
    """Integrate once in each time direction, using dense output at requested epochs."""
    seconds=np.asarray(seconds,float)
    def fun(t,y):
        r=y[:3]; d=np.linalg.norm(r)
        return np.r_[y[3:],-MU*r/d**3]
    out=np.empty((len(seconds),6))
    for sign,sel in [(1,seconds>=0),(-1,seconds<0)]:
        if not np.any(sel):continue
        tt=seconds[sel]; end=float(np.max(tt) if sign==1 else np.min(tt))
        if end==0:out[sel]=state;continue
        sol=solve_ivp(fun,(0.,end),state,method='DOP853',rtol=rtol,atol=atol,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        out[sel]=sol.sol(tt).T
    return out


@dataclass
class FitResult:
    epoch: np.datetime64
    circular_state: np.ndarray
    fitted_state: np.ndarray
    circular_chi2: float
    fitted_chi2: float
    n_observations: int
    circular_trials: int
    fit_trials: int
    success: bool
    message: str


def fit_orbit(times, ra_deg, dec_deg, lat_deg, lon_deg, altitude_km=0.,
              sigma_arcsec=5., range_bounds_km=(300.,60000.),
              circular_ranges=9, circular_phases=4, max_nfev=200,
              min_perigee_alt_km=100., seed=12, extra_seeds=5):
    """Fit a common Cartesian state to all passes, with multiple circular initializations.

    The best *physical* free-orbit solution is returned. Results depend on
    the assumed two-body dynamics and station/astrometry models.
    """
    t=np.asarray(times,dtype='datetime64[ns]'); ra=np.asarray(ra_deg,float); dec=np.asarray(dec_deg,float)
    if len(t)<6 or len(ra)!=len(t) or len(dec)!=len(t):raise ValueError('Need >=6 matched observations')
    order=np.argsort(t);t=t[order];ra=ra[order];dec=dec[order]
    epoch=t[len(t)//2]; dt=np.asarray((t-epoch)/np.timedelta64(1,'s'),float)
    stations=np.array([station_state(ti,lat_deg,lon_deg,altitude_km)[0] for ti in t])
    sr,_=station_state(epoch,lat_deg,lon_deg,altitude_km)
    obsra=np.deg2rad(ra);obsdec=np.deg2rad(dec)
    # Initial sightline interpolated from nearby observations (unwrapped RA).
    alpha=np.interp(0.,dt,np.unwrap(obsra)); delta=np.interp(0.,dt,obsdec)
    u=np.array([np.cos(delta)*np.cos(alpha),np.cos(delta)*np.sin(alpha),np.sin(delta)])
    def residual(s):
        pred=propagate_at(s,dt)[:,:3]-stations
        aa=np.arctan2(pred[:,1],pred[:,0]); dd=np.arctan2(pred[:,2],np.hypot(pred[:,0],pred[:,1]))
        da=np.arctan2(np.sin(aa-obsra),np.cos(aa-obsra))
        return np.column_stack((da,dd-obsdec)).ravel()*ARCSEC/sigma_arcsec
    def physical(s):
        ok,_=orbital_filter(s[:3][None,:],s[3:][None,:],min_perigee_alt_km)
        return bool(ok[0])
    def circular(logrho,phase):
        r=sr+np.exp(logrho)*u; nr=r/np.linalg.norm(r)
        e1=np.cross([0.,0.,1.],nr)
        if np.linalg.norm(e1)<1e-7:e1=np.cross([1.,0.,0.],nr)
        e1/=np.linalg.norm(e1);e2=np.cross(nr,e1)
        v=np.sqrt(MU/np.linalg.norm(r))*(np.cos(phase)*e1+np.sin(phase)*e2)
        return np.r_[r,v]
    low,high=np.log(range_bounds_km)
    starts=[]
    for logrho in np.linspace(low,high,circular_ranges):
        for phase in np.linspace(-np.pi,np.pi,circular_phases,endpoint=False):
            s=circular(logrho,phase)
            starts.append((float(np.dot(residual(s),residual(s))),s))
    starts.sort(key=lambda z:z[0]); rng=np.random.default_rng(seed)
    # Optimize circular parameters from several widely separated seeds.
    circular_solutions=[]
    for _,s in starts[:min(len(starts),max(8,extra_seeds*3))]:
        r=s[:3]; rho=np.linalg.norm(r-sr); v=s[3:]
        nr=r/np.linalg.norm(r); e1=np.cross([0.,0.,1.],nr)
        if np.linalg.norm(e1)<1e-7:e1=np.cross([1.,0.,0.],nr)
        e1/=np.linalg.norm(e1);e2=np.cross(nr,e1)
        phase=np.arctan2(np.dot(v,e2),np.dot(v,e1))
        opt=least_squares(lambda x:residual(circular(*x)),[np.log(rho),phase],
            bounds=([low,-2*np.pi],[high,2*np.pi]),max_nfev=max_nfev,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        ss=circular(*opt.x)
        if physical(ss):circular_solutions.append((np.dot(opt.fun,opt.fun),ss))
    if not circular_solutions:raise RuntimeError('No physical circular solution found')
    circular_solutions.sort(key=lambda z:z[0]); cchi,cs=circular_solutions[0]
    # Work in scaled Cartesian corrections to avoid km / km/s conditioning.
    # Scale around the first candidate; each trial has its own reference state.
    fitted=[]
    candidates=circular_solutions[:max(1,extra_seeds)]
    for _,ss in candidates:
        scales=np.array([1000.,1000.,1000.,1.,1.,1.])
        def fun(x):return residual(ss+x*scales)
        opt=least_squares(fun,np.zeros(6),method='trf',max_nfev=max_nfev,
            x_scale='jac',diff_step=1e-5,xtol=1e-10,ftol=1e-10,gtol=1e-10)
        state=ss+opt.x*scales
        if physical(state):fitted.append((np.dot(opt.fun,opt.fun),state,opt.success,opt.message))
    if not fitted:
        return FitResult(epoch,cs,cs,cchi,cchi,len(t),len(circular_solutions),len(candidates),False,'No physical free fit; circular result returned')
    fitted.sort(key=lambda z:z[0]);chi,s,success,msg=fitted[0]
    return FitResult(epoch,cs,s,float(cchi),float(chi),len(t),len(circular_solutions),len(candidates),bool(success),str(msg))
