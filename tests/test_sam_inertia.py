"""Physical inertia and actual contact updates of the production SAM path."""
import numpy as np
import pytest

from cufhom.pipeline.sam_generator import SAMLiteGenerator
from cufhom.pipeline.sam_paper_reference import rotational_inertia, sphere_exp
from cufhom.pipeline.sam_periodic_audit import segment_distance


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
    assert SAMLiteGenerator.rotational_scale(length, diameter) == pytest.approx(expected)
    assert rotational_inertia(length, diameter) == pytest.approx(expected)


def eccentric_pair():
    g = SAMLiteGenerator(np.eye(3)/3, 30., 1., .01, (60., 60., 60.),
                         tau_rotate=.03, num_threads=1)
    g._n_fibers = 2
    g._centers[:2] = [[30., 30., 30.], [36., 33., 30.6]]
    g._directions[:2] = [[1., 0., 0.], [0., 1., 0.]]
    return g


def test_production_contact_rotation_matches_energy_derivative():
    g = eccentric_pair()
    pairs = np.array([[0, 1]], dtype=np.int32)
    moves, rotations = np.zeros((2, 3)), np.zeros((2, 3))
    g._apply_pair_forces_vec(pairs, moves, rotations, .3)
    delta = g.centers[1]-g.centers[0]

    def energy(directions):
        distance = segment_distance(delta, 30*directions[0], 30*directions[1])
        return .5*max(0., 1.-float(distance))**2

    # Both closest points are away from the centroids, so contact torques
    # are nonzero. Check each tangent component through the public force path.
    eps = integrated_capsule_inertia(30., 1.)
    h = 1e-6
    for i in range(2):
        for axis in range(3):
            tangent = np.zeros((2, 3))
            tangent[i, axis] = 1.
            tangent -= g.directions*np.sum(g.directions*tangent, axis=1)[:, None]
            derivative = (energy(sphere_exp(g.directions, h*tangent))
                          -energy(sphere_exp(g.directions, -h*tangent)))/(2*h)
            update = float(np.sum(rotations*tangent))
            assert update == pytest.approx(-g.tau_rotate*derivative/eps, abs=1e-10)
    assert np.linalg.norm(rotations) > 0


def test_fire_compensated_contact_rotation_is_preserved():
    g = eccentric_pair()
    pairs = np.array([[0, 1]], dtype=np.int32)
    current_eps = g._rot_scale
    ar = 30.
    previous_eps = 2/3+ar*(ar**3/12+ar**2/6+3*ar/16+1/15)
    updates = []
    for eps in (previous_eps, current_eps):
        g._rot_scale = eps
        gain = max(g.tau_rotate, 2*eps/g.fiber_length**2)
        moves, rotations = np.zeros((2, 3)), np.zeros((2, 3))
        g._apply_pair_forces_vec(pairs, moves, rotations, .3, tau_rotate_override=gain)
        updates.append(rotations)
    np.testing.assert_allclose(*updates, rtol=1e-14, atol=1e-15)
