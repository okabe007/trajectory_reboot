
from __future__ import annotations
import numpy as np, math
from typing import Iterator, Tuple, Dict

# ---------------- CUBE -----------------
def cube_scenarios(step: float, radius: float) -> Iterator[Tuple[str, Dict]]:
    faces = {
        'top':    np.array([0,0, 1.]),
        'bottom': np.array([0,0,-1.]),
        'front':  np.array([0, 1.,0]),
        'back':   np.array([0,-1.,0]),
        'right':  np.array([1.,0,0]),
        'left':   np.array([-1.,0,0]),
    }
    inside = radius - 1.5*step
    for fname,n in faces.items():
        start = n*inside
        yield f"cube_{fname}", dict(
            start_position=start.tolist(),
            first_temp=(start+n*0.8*step).tolist(),
            reflection_vector=(n*step).tolist(),
            shape='cube'
        )
    tang = {
        'top':    (np.array([1,0,0]), np.array([0,1,0])),
        'bottom': (np.array([1,0,0]), np.array([0,1,0])),
        'front':  (np.array([1,0,0]), np.array([0,0,1])),
        'back':   (np.array([1,0,0]), np.array([0,0,1])),
        'right':  (np.array([0,1,0]), np.array([0,0,1])),
        'left':   (np.array([0,1,0]), np.array([0,0,1])),
    }
    for fname,(t1,t2) in tang.items():
        for idx,t in enumerate((t1,t2),1):
            u=t/np.linalg.norm(t)
            start = faces[fname]*inside
            yield f"cube_{fname}_edge{idx}", dict(
                start_position=start.tolist(),
                first_temp=(start+u*0.8*step).tolist(),
                reflection_vector=(u*step).tolist(),
                shape='cube'
            )

# ---------------- DROP ------------------
def drop_scenarios(step: float, R: float, theta: float):
    inside = R - 1.2*step
    dirs = {
        'xp': np.array([1,0,0]),
        'xm': np.array([-1,0,0]),
        'yp': np.array([0,1,0]),
        'ym': np.array([0,-1,0]),
        'zp': np.array([0,0,1]),
        'zm': np.array([0,0,-1]),
    }
    for key,d in dirs.items():
        u=d/np.linalg.norm(d)
        start=u*inside
        yield f"drop_{key}", dict(
            start_position=start.tolist(),
            first_temp=(start+u*0.8*step).tolist(),
            reflection_vector=(u*step).tolist(),
            shape='drop'
        )

# ---------------- SPOT (10) -------------
def spot_scenarios(step: float, R: float, theta: float):
    sinθ, cosθ = math.sin(theta), math.cos(theta)
    r_base = R*sinθ
    H = R*(1-cosθ)

    def make(name,start,v):
        yield name, dict(
            start_position=start.tolist(),
            first_temp=(start+v*0.8*step).tolist(),
            reflection_vector=(v*step).tolist(),
            shape='spot'
        )

    for vec,lab in [(np.r_[1,0,0],'xp'),(-np.r_[1,0,0],'xm'),(np.r_[0,1,0],'yp'),(-np.r_[0,1,0],'ym')]:
        u=vec/np.linalg.norm(vec)
        start=u*(r_base-1.2*step)
        yield from make(f"spot_A_{lab}",start,u)

    z_mid=H/2
    r_mid=math.sqrt(R*R-(z_mid-R*cosθ)**2)
    for vec,lab in [(np.r_[1,0,0],'xp'),(-np.r_[1,0,0],'xm'),(np.r_[0,1,0],'yp'),(-np.r_[0,1,0],'ym')]:
        u=vec/np.linalg.norm(vec)
        start=u*(r_mid-1.2*step)+np.r_[0,0,z_mid]
        yield from make(f"spot_B_{lab}",start,u)

    start=np.r_[0,0,H-1.3*step]
    yield from make("spot_C_up",start,np.r_[0,0,1])

    # F mid-height → diag-down +Y 方向のみ (1 本)
    vec, lab = np.r_[0,1,-1], 'yp'
    d=vec/np.linalg.norm(vec)
    start=np.r_[0,0,z_mid]+d*1.3*step
    yield from make(f"spot_F_{lab}",start,d)
