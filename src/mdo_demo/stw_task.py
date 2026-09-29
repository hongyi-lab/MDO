"""Physical utilities for the official STW fixed-planform task (SI units).

Frame: x downstream, y right semispan, z up. No CRM frame offsets.
The formulas keep TSFC in kg/(N s), so Breguet exponents include gravity.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

G = 9.80665


def read_stw_bdf(path):
    nodes, elems, labels = {}, [], {}
    lines = Path(path).read_text().splitlines()
    family = None
    for i, line in enumerate(lines):
        if 'Shell element data for family' in line:
            family = line.split('family', 1)[1].strip()
        if line.startswith('GRID*'):
            nid = int(line[8:24]); nodes[nid] = [float(line[40:56]), float(line[56:72]), float(lines[i+1][8:24])]
        elif line.startswith('CQUAD4'):
            e = [int(line[j:j+8]) for j in range(8, 56, 8)]
            elems.append(e); labels[e[1]] = family
    ids = sorted(nodes); lookup = {v:i for i,v in enumerate(ids)}
    xyz = np.array([nodes[i] for i in ids])
    prop = np.array([e[1] for e in elems])
    conn = np.array([[lookup[n] for n in e[2:]] for e in elems])
    if len(ids) < 100 or not all(labels.values()):
        raise ValueError('Expected official STW GRID*/CQUAD4 mesh with family labels')
    return dict(ids=np.array(ids), nodes=xyz, conn=conn, prop=prop, labels=labels)


def section_change(points, surface, twist, thickness=(1.,1.,1.),camber_delta=(0.,0.,0.)):
    """LE-preserving twist and symmetric thickness scaling about the camber line.

    Thickness controls vary along span; the baseline camber line is preserved.
    """
    p = np.asarray(points).copy(); s = np.asarray(surface)
    le = s[np.arange(len(s)), s[:,:,0].argmin(axis=1)]
    y = p[...,1]; eta = y/14.
    xle = np.interp(y, le[:,1], le[:,0]); zle = np.interp(y, le[:,1], le[:,2])
    t = np.deg2rad(np.interp(eta,[0.,.5,1.],[0.,*twist]))
    scale = np.interp(eta,[0.,.5,1.],thickness)
    # All baseline sections are geometrically similar RAE2822 sections.
    root=s[0]; leidx=int(root[:,0].argmin()); chord=float(root[:,0].max()-root[:,0].min())
    lo=root[:leidx+1][::-1]; up=root[leidx:255]
    frac=(p[...,0]-xle)/(5.-.25*y)
    loz=np.interp(frac,(lo[:,0]-root[leidx,0])/chord,(lo[:,2]-root[leidx,2])/chord)
    upz=np.interp(frac,(up[:,0]-root[leidx,0])/chord,(up[:,2]-root[leidx,2])/chord)
    camber=zle+.5*(loz+upz)*(5.-.25*y)
    # Zero value and slope at LE/TE: no hidden section rotation or LE-radius change.
    bump=16.*frac**2*(1.-frac)**2
    camber_shift=np.interp(eta,[0.,.5,1.],camber_delta)*(5.-.25*y)*bump
    dx = p[...,0]-xle; dz = camber-zle+(p[...,2]-camber)*scale+camber_shift
    p[...,0] = xle+np.cos(t)*dx+np.sin(t)*dz
    p[...,2] = zle-np.sin(t)*dx+np.cos(t)*dz
    return p


def atmosphere(altitude):
    T = 288.15-.0065*altitude
    p = 101325.*(T/288.15)**(G/(287.05287*.0065))
    return dict(T=T, rho=p/(287.05287*T), a=np.sqrt(1.4*287.05287*T))


def mission(mass_box, cl, cd):
    if min(mass_box,cl,cd) <= 0 or not np.isfinite([mass_box,cl,cd]).all():
        raise ValueError('Positive finite cruise mass/CL/CD required')
    wing = 10.147*mass_box**.8162
    lgm = 41500.+2*wing
    speed = .77*atmosphere(10400.)['a']
    ld = cl/(cd+.01508)
    gamma = np.arctan(10400./(180.*1609.34))
    cruise_start = lgm*np.exp(3815e3*18.1e-6*G/(speed*ld))
    togm = cruise_start*np.exp((180.*1609.34)*18.1e-6*G/(350./2.25)*(np.cos(gamma)/ld+np.sin(gamma)))
    return dict(wing_mass_kg=wing, LGM_kg=lgm, cruise_start_kg=cruise_start,
                mid_cruise_kg=np.sqrt(lgm*cruise_start), TOGM_kg=togm,
                fuel_burn_kg=togm-lgm, total_fuel_kg=togm-lgm+2000., L_D=ld)


class Transfer:
    """Conservative local force/couple map and its virtual-work transpose."""
    def __init__(self, nodes, points, neighbors=4):
        self.nodes=np.asarray(nodes); self.points=np.asarray(points)
        d,self.ids=cKDTree(nodes).query(points,k=neighbors)
        w=1/np.maximum(d,1e-10); self.w=w/w.sum(axis=1,keepdims=True)
        self.arm=self.points[:,None]-self.nodes[self.ids]

    def loads(self, forces):
        n=np.zeros((len(self.nodes),6)); f=self.w[...,None]*forces[:,None]
        m=np.cross(self.arm,f)
        for k in range(self.ids.shape[1]):
            np.add.at(n[:,:3],self.ids[:,k],f[:,k]); np.add.at(n[:,3:],self.ids[:,k],m[:,k])
        source=np.r_[np.sum(forces,axis=0),np.sum(np.cross(self.points,forces),axis=0)]
        target=np.r_[n[:,:3].sum(axis=0),(np.cross(self.nodes,n[:,:3])+n[:,3:]).sum(axis=0)]
        if not np.allclose(source,target,rtol=1e-10,atol=1e-7): raise RuntimeError('Load conservation failed')
        return n

    def motion(self, u):
        a=u[self.ids]
        return np.sum(self.w[...,None]*(a[...,:3]+np.cross(a[...,3:],self.arm)),axis=1)


def wingbox_volume(mesh, nodes):
    """Enclosed gross volume from outer panels plus two end ribs, not internal ribs."""
    vol=0.
    xyz=np.asarray(nodes)
    for conn,pid in zip(mesh['conn'],mesh['prop']):
        name=mesh['labels'][pid]; p=xyz[conn]
        if name.startswith('RIBS') and not ('RIB.00/' in name or 'RIB.22/' in name): continue
        if name.startswith('U_SKIN'): axis,sign=2,1
        elif name.startswith('L_SKIN'): axis,sign=2,-1
        elif name.startswith('RIBS'): axis,sign=1,(-1 if 'RIB.00/' in name else 1)
        else: axis,sign=0,(-1 if 'SPAR.00/' in name else 1)
        normal=np.cross(p[1]-p[0],p[2]-p[0])+np.cross(p[2]-p[0],p[3]-p[0])
        s=1 if normal[axis]*sign>0 else -1
        vol+=s*(np.dot(p[0],np.cross(p[1],p[2]))+np.dot(p[0],np.cross(p[2],p[3])))/6
    if not np.isfinite(vol) or vol<=0: raise ValueError('Invalid enclosed wingbox volume')
    return float(vol)


def sizing_margins(design,panels):
    """Independent official sizing/adjacency checks, all returned in metres."""
    d=np.asarray(design); a=np.array([d[r['dv_start']:r['dv_start']+3] for r in panels]); t,h,ts=a.T
    c={'min_panel_thickness':float((t-.0006).min()),'min_stiffener_thickness':float((ts-.0006).min()),
       'min_height_and_flange_width':float((h-.0254).min()),'max_stiffener_thickness':float((15*t-ts).min()),
       'max_stiffener_aspect':float((30*ts-h).min()),'min_stiffener_aspect':float((h-5*ts).min()),
       'stiffener_spacing':float((.15-h).min())}
    adj=[]
    for family in ('U_SKIN','L_SKIN','SPARS/SPAR.00','SPARS/SPAR.01'):
        rows=[i for i,r in enumerate(panels) if r['name'].startswith(family)]
        rows.sort(key=lambda i:panels[i]['centroid'][1]); adj.extend((i,j) for i,j in zip(rows[:-1],rows[1:]))
    delta=np.array([abs(a[i]-a[j]) for i,j in adj])
    for k,name in enumerate(('panel','height','stiffener')):c['adjacent_'+name]=float((np.array([.0025,.01,.0025])[k]-delta[:,k]).min())
    return c
