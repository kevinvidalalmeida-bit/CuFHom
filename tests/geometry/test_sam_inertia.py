"""Physical inertia and actual contact updates of the production SAM path."""
import numpy as np
import pytest

from cufhom.geometry.contacts import contact_energy
from cufhom.geometry.contacts import rotational_inertia, sphere_exp
from cufhom.geometry.audit import segment_distance


def integrated_capsule_inertia(length, diameter):
    """Integrate disk areas/moments along the capsule, independently of AR formula."""
    nodes, weights = np.polynomial.legendre.leggauss(16)
    radius = diameter/2
    z = length*nodes/2
    dz = length*weights/2
    volume = np.sum(np.pi*radius**2*dz)
    inertia = np.sum((np.pi*radius**4/4+z*z*np.pi*radius**2)*dz)
    t = radius*(nodes+1)/2
    dt = radius*weights/2
    radius2 = radius**2-t*t
    cap_z = length/2+t
    volume += 2*np.sum(np.pi*radius2*dt)
    inertia += 2*np.sum((np.pi*radius2**2/4+cap_z**2*np.pi*radius2)*dt)
    return inertia/volume


@pytest.mark.parametrize('ar,diameter', [(1., 1.), (5., .2), (30., 1.), (100., 10.)])
def test_production_inertia_matches_volume_integral(ar, diameter):
    length = ar*diameter
    expected = integrated_capsule_inertia(length, diameter)
    assert rotational_inertia(length, diameter) == pytest.approx(expected)


def test_production_contact_rotation_matches_energy_derivative():
    c = np.array([[30.,30.,30.], [36.,33.,30.6]])
    u = np.array([[1.,0.,0.], [0.,1.,0.]])
    _,_,gu,_ = contact_energy(c,u,30.,1.,np.full(3,60.))
    def energy(directions):
        distance = segment_distance(c[1]-c[0], 30*directions[0], 30*directions[1])
        return .5*max(0.,1.-float(distance))**2
    for i in range(2):
        for axis in range(3):
            tangent = np.zeros((2,3)); tangent[i,axis] = 1.
            tangent -= u*np.sum(u*tangent,axis=1)[:,None]
            derivative = (energy(sphere_exp(u,1e-6*tangent))-energy(sphere_exp(u,-1e-6*tangent)))/2e-6
            assert np.sum(gu*tangent) == pytest.approx(derivative,abs=1e-9)
    assert np.linalg.norm(gu)>0
