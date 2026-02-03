"""
Test decay model fitting to ensure it works correctly with lmfit and fitspy.
"""

import numpy as np
import pytest
from lmfit import Model
from lmfit.models import LinearModel

from spectroview.model.m_fit_models import decay_single_exp, decay_bi_exp


@pytest.fixture
def synthetic_decay_data():
    """Generate synthetic TRPL decay data for testing."""
    np.random.seed(42)
    t = np.linspace(0, 40, 400)
    A_true = 1200
    tau_true = 0.26
    B_true = 768
    
    y_true = A_true * np.exp(-t / tau_true) + B_true
    noise = np.random.normal(0, 20, len(t))
    y_data = y_true + noise
    
    return {
        't': t,
        'y': y_data,
        'A_true': A_true,
        'tau_true': tau_true,
        'B_true': B_true
    }


def test_decay_single_exp_direct_lmfit(synthetic_decay_data):
    """Test 1: Direct lmfit Model fitting for DecaySingleExp."""
    data = synthetic_decay_data
    
    # Create model
    model = Model(decay_single_exp, independent_vars=['x'])
    
    # Set initial parameters
    params = model.make_params()
    params['A'].set(value=np.max(data['y']), min=0)
    params['tau'].set(value=5.0, min=0.1, max=100)
    params['B'].set(value=np.min(data['y']), min=0)
    
    # Fit
    result = model.fit(data['y'], params, x=data['t'])
    
    # Assertions
    assert result.success, f"Fit failed: {result.message}"
    
    # Check fitted values are close to true values (within 10%)
    assert abs(result.params['A'].value - data['A_true']) / data['A_true'] < 0.1
    assert abs(result.params['tau'].value - data['tau_true']) / data['tau_true'] < 0.1
    assert abs(result.params['B'].value - data['B_true']) / data['B_true'] < 0.1


def test_decay_single_exp_fitspy_create_model(synthetic_decay_data):
    """Test 2: Fitspy's create_model approach for DecaySingleExp."""
    from fitspy.core.spectrum import create_model
    
    data = synthetic_decay_data
    
    # Create model with fitspy
    model = create_model(decay_single_exp, "DecaySingleExp", prefix="m01_")
    
    # Set parameter hints
    model.set_param_hint("A", value=np.max(data['y']), min=0)
    model.set_param_hint("tau", value=5.0, min=0.1, max=100)
    model.set_param_hint("B", value=np.min(data['y']), min=0)
    
    # Make params
    params = model.make_params()
    
    # Fit
    result = model.fit(data['y'], params, x=data['t'])
    
    # Assertions
    assert result.success, f"Fit failed: {result.message}"
    
    # Check fitted values (with prefix)
    assert abs(result.params['m01_A'].value - data['A_true']) / data['A_true'] < 0.1
    assert abs(result.params['m01_tau'].value - data['tau_true']) / data['tau_true'] < 0.1
    assert abs(result.params['m01_B'].value - data['B_true']) / data['B_true'] < 0.1


def test_decay_with_baseline_composite(synthetic_decay_data):
    """Test 3: Decay model + linear baseline composite."""
    data = synthetic_decay_data
    
    # Create composite model
    decay_model = Model(decay_single_exp, independent_vars=['x'], prefix='decay_')
    baseline_model = LinearModel(prefix='baseline_')
    composite = decay_model + baseline_model
    
    # Set initial parameters
    params = composite.make_params()
    params['decay_A'].set(value=np.max(data['y']), min=0)
    params['decay_tau'].set(value=5.0, min=0.1, max=100)
    params['decay_B'].set(value=np.min(data['y']), min=0)
    params['baseline_slope'].set(value=0, vary=True)
    params['baseline_intercept'].set(value=0, vary=True)
    
    # Fit
    result = composite.fit(data['y'], params, x=data['t'])
    
    # Assertions
    assert result.success, f"Fit failed: {result.message}"
    
    # Check decay parameters are still reasonably fitted
    # (baseline should absorb some offset, so be more lenient)
    assert abs(result.params['decay_tau'].value - data['tau_true']) / data['tau_true'] < 0.2


def test_decay_bi_exp_direct_lmfit(synthetic_decay_data):
    """Test 4: BiExponential decay model fitting."""
    data = synthetic_decay_data
    
    # Create bi-exp model
    model = Model(decay_bi_exp, independent_vars=['x'])
    
    # Set initial parameters
    params = model.make_params()
    params['A1'].set(value=np.max(data['y']) * 0.7, min=0)
    params['tau1'].set(value=2.0, min=0.1, max=50)
    params['A2'].set(value=np.max(data['y']) * 0.3, min=0)
    params['tau2'].set(value=10.0, min=0.1, max=100)
    params['B'].set(value=np.min(data['y']), min=0)
    
    # Fit (should succeed even if data is single-exp)
    result = model.fit(data['y'], params, x=data['t'])
    
    # Assertion - fit should succeed
    assert result.success, f"Bi-exp fit failed: {result.message}"
