# dory/sk.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
import logging

logger = logging.getLogger("dory.sk")
logger.addHandler(logging.NullHandler())

@dataclass
class SKParams:
    """Heteropolar Slater–Koster parameters for A→B (AB) and B→A (BA)."""
    AB: Dict[str, float]
    BA: Dict[str, float]

def _pick_params(p: Union[Dict[str,float], SKParams], orientation: str) -> Dict[str,float]:
    if isinstance(p, dict):
        return p
    if orientation not in ("AB","BA"):
        raise ValueError("orientation must be 'AB' or 'BA'")
    return p.AB if orientation == "AB" else p.BA

def direction_cosines(delta: np.ndarray) -> tuple[float, float, float, float]:
    """Return (l, m, n, |delta|) for a bond vector δ."""
    d = np.asarray(delta, dtype=float).reshape(3)
    R = float(np.linalg.norm(d))
    if R == 0.0:
        logger.error("Zero bond length for delta=%s", delta)
        raise ValueError("Zero bond length.")
    l, m, n = (d / R).tolist()
    logger.debug("δ=%s -> (l,m,n)=(%.6f, %.6f, %.6f), |δ|=%.6f", delta, l, m, n, R)
    return l, m, n, R

def sk_block_sp3d5(
    basis: List[str],
    delta: np.ndarray,
    params_or_sk: Union[Dict[str, float], SKParams],
    orientation: str = "AB",
) -> np.ndarray:
    """
    Full single-bond Slater–Koster block M^(orientation)(hat{δ}) for a site-local basis
    ordered as (s; px,py,pz; dxy,dyz,dxz,dx2-y2,dz2). The block maps A→B.
    Accepts a raw dict of SK numbers or SKParams(AB, BA).
    """
    params = _pick_params(params_or_sk, orientation)
    l, m, n, _ = direction_cosines(delta)
    l2, m2, n2 = l*l, m*m, n*n
    dvec = np.array([l, m, n], float)
    sq3 = float(np.sqrt(3.0))
    delta_lm = l2 - m2
    t1 = n2 - 0.5*(l2 + m2)

    # pull parameters with 0.0 default
    g = lambda k: float(params.get(k, 0.0))
    V_ss, V_sp, V_ps  = g("ss_sigma"), g("sp_sigma"), g("ps_sigma")
    V_pps, V_ppp      = g("pp_sigma"), g("pp_pi")
    V_sd, V_ds        = g("sd_sigma"), g("ds_sigma")
    V_pds, V_pdp      = g("pd_sigma"), g("pd_pi")
    V_dps, V_dpp      = g("dp_sigma"), g("dp_pi")
    V_dds, V_ddp, V_ddd = g("dd_sigma"), g("dd_pi"), g("dd_delta")

    # parameters for s start orbitals 
    V_sstarsstar = g("sstar_sstar_sigma")
    V_sstars     = g("sstar_s_sigma")   
    V_sstarp     = g("sstar_p_sigma")
    V_psstar     = g("p_sstar_sigma")
    V_sstard     = g("sstar_d_sigma")
    V_dsstar     = g("d_sstar_sigma")

    nb = len(basis)
    idx = {o: i for i, o in enumerate(basis)}
    has = set(basis).__contains__
    M = np.zeros((nb, nb), dtype=np.complex128)

    # s–s
    if has("s"):
        M[idx["s"], idx["s"]] += V_ss

    # s-sstar
    if has("s*"):
        M[idx["s*"], idx["s*"]] += V_sstarsstar


    # s–p and p–s (minus from p parity on the bra side)
    pnames = [o for o in ("px","py","pz") if has(o)]
    if has("s") and pnames:
        s = idx["s"]
        comps = {"px": l, "py": m, "pz": n}
        for p in pnames:
            ip = idx[p]
            c = comps[p]
            M[s,  ip] +=  V_sp * c
            M[ip, s ] += -V_ps * c

    # s-sstar
    if has("s*") and has("s"):
        M[idx["s*"], idx["s"]] += V_sstars
        M[idx["s"], idx["s*"]] += V_sstars 

    # p -sstar 
    if has("s*") and pnames:
        sst = idx["s*"]
        comps = {"px": l, "py": m, "pz": n}
        for p in pnames:
            ip = idx[p]
            c = comps[p]
            M[sst, ip] += V_sstarp * c
            M[ip, sst] += -V_psstar * c   


    # p–p: (Vσ − Vπ) d d^T + Vπ I
    if len(pnames) == 3:
        P = [idx[p] for p in ("px","py","pz")]
        Mpp = (V_pps - V_ppp) * np.outer(dvec, dvec) + V_ppp * np.eye(3)
        M[np.ix_(P, P)] += Mpp
    elif pnames:
        comps = {"px": l, "py": m, "pz": n}
        for i, pi in enumerate(pnames):
            li = comps[pi]
            for j, pj in enumerate(pnames):
                lj = comps[pj]
                M[idx[pi], idx[pj]] += li*lj*(V_pps - V_ppp) + (V_ppp if i == j else 0.0)

    # s–d and d–s (d even → no extra sign)
    dnames = [o for o in ("dxy","dyz","dxz","dx2-y2","dz2") if has(o)]
    if has("s") and dnames:
        s = idx["s"]
        coeff = {
            "dxy":    sq3*l*m,
            "dyz":    sq3*m*n,
            "dxz":    sq3*l*n,
            "dx2-y2": 0.5*sq3*delta_lm,
            "dz2":    t1,
        }
        for d in dnames:
            c = coeff[d]
            M[s,      idx[d]] += V_sd * c
            M[idx[d], s     ] += V_ds * c

    # sstar - d
    if has("s*") and dnames:
        sst = idx["s*"]
        coeff = {
            "dxy":    sq3*l*m,
            "dyz":    sq3*m*n,
            "dxz":    sq3*l*n,
            "dx2-y2": 0.5*sq3*delta_lm,
            "dz2":    t1,
        }
        for d in dnames:
            c = coeff[d]
            M[sst,    idx[d]] += V_sstard * c
            M[idx[d], sst   ] += V_dsstar * c


    # helper to build a p–d angular block for given (Vσ,Vπ)
    def pd_block(Vsig: float, Vpi: float) -> np.ndarray:
        B = np.zeros((3,5), float)
        # px
        B[0,0] = m*(sq3*l2*Vsig + (1 - 2*l2)*Vpi)
        B[0,1] = l*m*n*(sq3*Vsig - 2*Vpi)
        B[0,2] = n*(sq3*l2*Vsig + (1 - 2*l2)*Vpi)
        B[0,3] = l*((0.5*sq3*delta_lm)*Vsig + (1 - delta_lm)*Vpi)
        B[0,4] = l*(t1*Vsig - sq3*n2*Vpi)
        # py
        B[1,0] = l*(sq3*m2*Vsig + (1 - 2*m2)*Vpi)
        B[1,1] = n*(sq3*m2*Vsig + (1 - 2*m2)*Vpi)
        B[1,2] = l*m*n*(sq3*Vsig - 2*Vpi)
        B[1,3] = m*((0.5*sq3*delta_lm)*Vsig - (1 + delta_lm)*Vpi)
        B[1,4] = m*(t1*Vsig - sq3*n2*Vpi)
        # pz
        B[2,0] = l*m*n*(sq3*Vsig - 2*Vpi)
        B[2,1] = l*(sq3*n2*Vsig + (1 - 2*n2)*Vpi)
        B[2,2] = m*(sq3*n2*Vsig + (1 - 2*n2)*Vpi)
        B[2,3] = n*((0.5*sq3*delta_lm)*Vsig - (delta_lm)*Vpi)
        B[2,4] = n*(t1*Vsig + sq3*(l2 + m2)*Vpi)
        return B

    # p–d and d–p (minus on d–p from p parity)
    if pnames and dnames:
        prows = ["px","py","pz"]
        dcols = ["dxy","dyz","dxz","dx2-y2","dz2"]
        P = pd_block(V_pds, V_pdp)  # pd with (pdσ,pdπ)
        D = pd_block(V_dps, V_dpp)  # dp uses same angulars with (dpσ,dpπ)
        for ip, p in enumerate(prows):
            if not has(p): continue
            for jd, d in enumerate(dcols):
                if not has(d): continue
                M[idx[p], idx[d]] +=  P[ip, jd]
                M[idx[d], idx[p]] += -D[ip, jd]

    # d–d block (fill upper triangle and assign the transpose)
    if dnames:
        def put(i: str, j: str, val: float) -> None:
            if has(i) and has(j):
                M[idx[i], idx[j]] += val

        # diagonals
        put("dxy","dxy", 3*l2*m2*V_dds + (l2 + m2 - 4*l2*m2)*V_ddp + (n2 + l2*m2)*V_ddd)
        put("dyz","dyz", 3*m2*n2*V_dds + (m2 + n2 - 4*m2*n2)*V_ddp + (l2 + m2*n2)*V_ddd)
        put("dxz","dxz", 3*l2*n2*V_dds + (l2 + n2 - 4*l2*n2)*V_ddp + (m2 + l2*n2)*V_ddd)
        put("dx2-y2","dx2-y2", 0.75*delta_lm**2*V_dds + (l2 + m2 - delta_lm**2)*V_ddp + (n2 + 0.25*delta_lm**2)*V_ddd)
        put("dz2","dz2", (t1**2)*V_dds + 3*n2*(l2 + m2)*V_ddp + 0.75*(l2 + m2)**2*V_ddd)

        # t2g pairs
        put("dxy","dyz", (l*n)*(3*m2*V_dds + (1 - 4*m2)*V_ddp + (m2 - 1)*V_ddd))
        put("dxy","dxz", (m*n)*(3*l2*V_dds + (1 - 4*l2)*V_ddp + (l2 - 1)*V_ddd))
        put("dyz","dxz", (l*m)*(3*n2*V_dds + (1 - 4*n2)*V_ddp + (n2 - 1)*V_ddd))

        # t2g–eg
        def t2g_dx2y2(prod, p0, p1, d0, d1):
            return prod*(1.5*delta_lm*V_dds + (p0 + p1*delta_lm)*V_ddp + (d0 + d1*delta_lm)*V_ddd)
        put("dxy","dx2-y2", t2g_dx2y2(l*m,  0.0, -2.0,  0.0, 0.5))
        put("dyz","dx2-y2", t2g_dx2y2(n*m, -1.0, -2.0,  1.0, 0.5))
        put("dxz","dx2-y2", t2g_dx2y2(l*n,  1.0, -2.0, -1.0, 0.5))

        def t2g_dz2(prod, P, Dv):
            return sq3*prod*(t1*V_dds + P*V_ddp + Dv*V_ddd)
        put("dxy","dz2", t2g_dz2(l*m, -2.0*n2,  0.5*(1.0 + n2)))
        put("dyz","dz2", t2g_dz2(n*m,  (l2 + m2 - n2), -0.5*(l2 + m2)))
        put("dxz","dz2", t2g_dz2(l*n,  (l2 + m2 - n2), -0.5*(l2 + m2)))

        # eg–eg
        put("dx2-y2","dz2", 0.5*sq3*delta_lm*t1*V_dds + sq3*n2*(m2 - l2)*V_ddp + 0.25*sq3*(1.0 + n2)*delta_lm*V_ddd)

        # symmetrize (assignment, not +=)
        dfull = ["dxy","dyz","dxz","dx2-y2","dz2"]
        for i in range(len(dfull)):
            for j in range(i+1, len(dfull)):
                di, dj = dfull[i], dfull[j]
                if has(di) and has(dj):
                    M[idx[dj], idx[di]] = M[idx[di], idx[dj]]

    logger.debug("Built SK block (orientation=%s) for basis size %d", orientation, nb)
    return M