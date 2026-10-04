"""Independent algebra and support controls for the paper's added analyses."""
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from paper.numerical_audit.matched_analysis import analyze, symmetric_split


def test_symmetric_split_known_normalization_reversal():
    own, other = symmetric_split(2., 8., 1., 1.)
    # Average of 2/10 -> 1/9 and 2/3 -> 1/2 paths.
    assert own == pytest.approx(((1/9-1/5)+(1/2-2/3))/2)
    assert own < 0 < other
    assert own + other == pytest.approx(.3)


def test_split_one_factor_changes_and_swap_symmetry():
    own, other = symmetric_split(2., 3., 4., 3.)
    assert own == pytest.approx(4/7-2/5)
    assert other == pytest.approx(0.)
    reverse_own, reverse_other = symmetric_split(4., 3., 2., 3.)
    assert reverse_own == pytest.approx(-own)
    assert reverse_other == pytest.approx(-other)
    a, b = symmetric_split(3., 2., 3., 4.)
    assert a == pytest.approx(-other)
    assert b == pytest.approx(-own)


def test_split_identity_random_positive_data():
    rng = np.random.default_rng(24)
    a0, b0, a1, b1 = np.exp(rng.normal(size=(4, 100)))
    own, other = symmetric_split(a0, b0, a1, b1)
    assert_allclose(own+other, a1/(a1+b1)-a0/(a0+b0), rtol=0, atol=3e-16)


@pytest.mark.parametrize('values', [(0, 1, 1, 0), (-1, 2, 1, 1), (np.nan, 1, 2, 1)])
def test_undefined_crossed_ratios_and_invalid_values_rejected(values):
    with pytest.raises(ValueError):
        symmetric_split(*values)


def test_common_width_support_removes_an_artificial_support_reversal():
    table = {}
    for width in (2, 5, 10):
        for geometry in ('g1', 'g2', 'g3'):
            if geometry == 'g3' and width == 10:
                continue
            before = np.ones(7)
            after = np.ones(7)
            after[0] = .001 if geometry == 'g3' else 1.1
            key = (geometry, 'L', 'baseline', 'L', width)
            table[(0, *key)] = before
            table[(100, *key)] = after
    result = analyze(table)
    rows = [r for r in result['common_width_comparisons'] if r['segment'] == 'L1']
    assert len(rows) == 3
    assert all(r['paired'] == r['n_geometry'] == 2 for r in rows)
    assert_allclose([r['equal_geometry_pp'] for r in rows], 100*(1.1/7.1-1/7))
    # The unmatched 2-mm mean is negative, despite a positive matched result.
    assert (2*(1.1/7.1-1/7)+(.001/6.001-1/7))/3 < 0
    json.dumps(result, allow_nan=False)


def test_stencil_perturbation_bound_is_sharp():
    epsilon, width = .003, .005
    perturbation = np.array([epsilon, -epsilon, epsilon])
    observed = np.diff(perturbation, n=2)[0]/width**2
    assert observed == pytest.approx(4*epsilon/width**2)


def test_continuous_affine_partial_bin_bias():
    width, remainder, slope = .005, .0015, 100.
    # Last two full bins plus one incomplete bin, using exact continuous means.
    centres = np.array([3.5*width, 4.5*width, 5*width+remainder/2])
    observed = np.diff(slope*centres, n=2)[0]/width**2
    expected = slope*(remainder-width)/(2*width**2)
    assert observed == pytest.approx(expected)
    assert observed == pytest.approx(-7000.)


def test_filter_multiplier_from_analytic_sinusoid_integrals():
    width, wavenumber = .005, 170.
    edges = np.arange(8)*width
    means = (np.cos(wavenumber*edges[:-1])-np.cos(wavenumber*edges[1:]))/(wavenumber*width)
    centres = (edges[:-1]+edges[1:])/2
    af = np.diff(means, n=2)/width**2
    factor = -4*np.sin(wavenumber*width/2)**2/width**2
    factor *= np.sinc(wavenumber*width/(2*np.pi))
    assert_allclose(af, factor*np.sin(wavenumber*centres[1:-1]), rtol=1e-13, atol=1e-10)
