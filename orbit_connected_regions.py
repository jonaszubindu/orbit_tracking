"""Optional connected-region diagnostic on Stage A's COARSE, fixed-attributable grid.

This is not a profiled likelihood or a proof of global connectivity.  It never
changes Stage A/B states or posterior weights.  When grid support is inadequate,
return a failed diagnostic so callers can use original family labels.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label
from orbit_toy import station_state

@dataclass
class ConnectedRegions:
    labels: np.ndarray
    grid_labels: np.ndarray
    n_components: int
    assigned_weight: float
    passed: bool
    reason: str
    threshold: float


def identify_regions(a, b, delta_chi2=16.812, min_weight_coverage=0.8,
                     connectivity=8, min_region_cells=10):
    """Label 4/8-connected cells inside a fixed-attributable chi2 contour.

    Threshold is min(finite grid chi2)+delta_chi2.  Candidate points must land
    *inside* a qualifying grid cell; off-grid candidates remain unassigned (0).
    This strict policy intentionally fails when the grid is too coarse or its
    fixed-attributable surface is a poor surrogate for the profiled likelihood.
    """
    ranges=np.asarray(a['ranges_km'],float); rates=np.asarray(a['rates_km_s'],float)
    grid=np.asarray(a['grid_chi2'],float); physical=np.asarray(a['grid_physical'],bool)
    states=np.asarray(b['states'],float);weights=np.asarray(b['weights'],float)
    if grid.shape!=(len(ranges),len(rates)) or states.ndim!=2 or states.shape[1]!=6:
        raise ValueError('Inconsistent Stage A grid or Stage B state dimensions')
    if not np.any(np.isfinite(grid)&physical):
        return ConnectedRegions(np.zeros(len(states),int),np.zeros_like(grid,int),0,0.,False,'No finite physical grid cells',np.inf)
    threshold=float(np.min(grid[physical&np.isfinite(grid)])+delta_chi2)
    mask=physical & np.isfinite(grid) & (grid<=threshold)
    structure=np.ones((3,3),int) if connectivity==8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    components,n=label(mask,structure)
    epoch=np.datetime64(a['epoch'],'ns')
    station=(float(a['station_lat_deg']),float(a['station_lon_deg']),float(a['station_alt_km']))
    sr,sv=station_state(epoch,*station)
    d=states[:,:3]-sr;rho=np.linalg.norm(d,axis=1)
    rate=np.einsum('ij,ij->i',states[:,3:]-sv,d)/np.maximum(rho,1e-12)
    # Only map to a cell if inside the cell's bounded Voronoi rectangle.
    lr=np.log(ranges); lq=np.log(np.maximum(rho,1e-100))
    ri=np.abs(lq[:,None]-lr).argmin(axis=1)
    vi=np.abs(rate[:,None]-rates).argmin(axis=1)
    valid=(rho>=ranges[0])&(rho<=ranges[-1])&(rate>=rates[0])&(rate<=rates[-1])
    labels=np.zeros(len(states),int);labels[valid]=components[ri[valid],vi[valid]]
    weights=weights/weights.sum()
    coverage=float(weights[labels>0].sum())
    n_cells=int(mask.sum())
    passed=bool(n>0 and n_cells>=min_region_cells and coverage>=min_weight_coverage)
    reason=(f'Grid-connected labeling accepted ({coverage:.1%} weight covered)' if passed else
            f'Fallback: grid has only {n_cells} qualifying cells (required {min_region_cells}); '
            f'coverage {coverage:.1%} (required {min_weight_coverage:.0%}); retain original Stage B labels')
    return ConnectedRegions(labels,components,int(n),coverage,passed,reason,threshold)
