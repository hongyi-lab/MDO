import numpy as np
from mdo_demo.stw_task import Transfer, mission, section_change


def test_transfer_preserves_wrench_and_virtual_work():
    rng=np.random.default_rng(923)
    nodes=rng.normal(size=(31,3)); points=rng.normal(size=(70,3)); forces=rng.normal(size=(70,3))
    t=Transfer(nodes,points); f=t.loads(forces); u=rng.normal(size=(31,6))
    np.testing.assert_allclose(f[:,:3].sum(0),forces.sum(0),atol=1e-12)
    np.testing.assert_allclose((np.cross(nodes,f[:,:3])+f[:,3:]).sum(0),np.cross(points,forces).sum(0),atol=1e-12)
    np.testing.assert_allclose(np.sum(f*u),np.sum(forces*t.motion(u)),atol=1e-12)


def test_breguet_units_reproduce_public_case2_qoi():
    # Public rounded inputs: box=704.54 kg, whole aircraft L/D=16.17,
    # fuel=10983.96 kg. Omitting g with kg/(N s) TSFC gives a ~10x error.
    r=mission(704.54,.7,.7/16.17-.01508)
    assert abs(r['fuel_burn_kg']/10983.96-1)<.001
    assert abs(r['LGM_kg']/45783.81-1)<1e-5
    assert r['total_fuel_kg']==r['fuel_burn_kg']+2000.


def test_fuel_decreases_with_drag_at_fixed_mass():
    assert mission(1000.,.7,.025)['fuel_burn_kg']<mission(1000.,.7,.04)['fuel_burn_kg']


def test_geometric_transform_preserves_root_twist_and_le():
    x=np.linspace(1.,0.,128); x=np.r_[x,x[-2::-1],1.,1.]
    z=np.r_[-.1*np.sin(np.linspace(0,np.pi,128)),.1*np.sin(np.linspace(0,np.pi,128))[1:],0.,0.]
    p=np.zeros((129,257,3)); y=np.linspace(0.,14.,129)
    p[:,:,0]=x[None]*(5.-.25*y[:,None])+.5*y[:,None]
    p[:,:,1]=y[:,None]; p[:,:,2]=z[None]*(5.-.25*y[:,None])
    q=section_change(p,p,[-2.,-3.],[1.,1.,1.])
    np.testing.assert_allclose(q[0],p[0],atol=1e-12)
    np.testing.assert_allclose(q[:,127],p[:,127],atol=1e-12)
    np.testing.assert_allclose(q[:,0],q[:,-1],atol=1e-12)


def test_mission_rejects_nonphysical_coefficients():
    import pytest
    with pytest.raises(ValueError):mission(500.,-.1,.02)


def test_camber_preserves_section_thickness_and_endpoints():
    x=np.r_[np.linspace(1.,0.,128),np.linspace(0.,1.,128)[1:],1.,1.]
    z=np.r_[-.1*np.sin(np.linspace(0,np.pi,128)),.1*np.sin(np.linspace(0,np.pi,128))[1:],0.,0.]
    y=np.linspace(0.,14.,129);p=np.zeros((129,257,3))
    p[:,:,0]=x[None]*(5.-.25*y[:,None])+.5*y[:,None]
    p[:,:,1]=y[:,None];p[:,:,2]=z[None]*(5.-.25*y[:,None])
    q=section_change(p,p,[0.,0.],[1.,1.,1.],[-.02,.01,.005])
    np.testing.assert_allclose(q[:,[0,127,254]],p[:,[0,127,254]],atol=1e-12)
    # Mirrored upper/lower chord indices get the same camber shift.
    np.testing.assert_allclose(q[:,:128,2]-q[:,254:126:-1,2],
                               p[:,:128,2]-p[:,254:126:-1,2],atol=1e-12)
    assert np.max(np.abs(q-p))>.02
