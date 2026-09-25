import numpy as np

from neuroez_c.task2.ngbr.neural_fragility import fit_linear_dynamics,stabilize_dynamics,structured_column_fragility


def test_fragility_is_deterministic_and_finite_for_stable_system():
    matrix=np.array([[.9,.0,.0,.0],[.2,.4,.0,.0],[.0,.1,.3,.0],[.0,.0,.1,.2]])
    first=structured_column_fragility(matrix,64);second=structured_column_fragility(matrix,64)
    assert np.all(np.isfinite(first)) and np.allclose(first,second)
    assert first[0]>first[-1]


def test_batched_resolvent_matches_direct_definition():
    matrix=np.array([[.7,.1,0,0],[0,.5,.2,0],[0,0,.4,.1],[.05,0,0,.3]]);angles=16;identity=np.eye(4,dtype=complex);best=np.full(4,np.inf)
    for theta in np.linspace(0,2*np.pi,angles,endpoint=False):
        resolvent=np.linalg.solve(np.exp(1j*theta)*identity-matrix,identity);best=np.minimum(best,1/np.linalg.norm(resolvent,axis=1))
    expected=1/(best+1e-8)
    assert np.allclose(structured_column_fragility(matrix,angles),expected,rtol=1e-6,atol=1e-7)


def test_dynamics_fit_and_stability_rescale():
    rng=np.random.default_rng(2);data=rng.normal(size=(4,100));matrix=fit_linear_dynamics(data)
    assert matrix.shape==(4,4)
    stable,changed,rho=stabilize_dynamics(np.eye(4)*1.2)
    assert changed and rho>=1 and max(abs(np.linalg.eigvals(stable)))<1


def test_fragility_failure_handling_is_nan():
    matrix=np.full((4,4),np.nan)
    try:
        value=structured_column_fragility(matrix,8)
    except np.linalg.LinAlgError:
        value=np.full(4,np.nan)
    assert np.isnan(value).all()
