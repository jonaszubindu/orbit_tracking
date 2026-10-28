import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_coverage import credible_caps

def test_caps_known_mass():
    u=np.array([[1.,0,0],[0.,1,0],[-1.,0,0]])
    w=np.array([.8,.15,.05]);r,c,sep=credible_caps(u,w,u[0]);assert np.all(c)
    assert r[0]==0 and r[1]>0 and r[2]>0

def test_caps_outside():
    u=np.array([[1.,0,0],[0.,1,0]])
    r,c,sep=credible_caps(u,np.array([.9,.1]),np.array([-1.,0,0]));assert not np.any(c)
