"""Scientific and leakage invariants on synthetic data; no CSS runs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom
from sklearn.model_selection import ParameterGrid

from h2psa import step6


def frame():
    data = pd.DataFrame(np.full((512, 8), .4), columns=step6.INPUTS)
    data['id'] = [f'D_train_{i:04d}' for i in range(512)]
    data['status'] = 'css'
    data['z'] = np.arange(512) / 512
    data['recovery'] = .5
    data['productivity'] = 10.
    data.loc[0, ['status', 'z', 'recovery', 'productivity']] = ['censored', 3., 999., 999.]
    data.loc[1, ['status', 'z', 'recovery', 'productivity']] = ['failed', np.nan, np.nan, np.nan]
    data.loc[2, 'status'] = 'cap'
    return data


def test_registered_models_grid_and_optimization_budget():
    assert len(step6.fit_tasks()) == 24
    assert len(ParameterGrid(step6.GBM_GRID)) == 54
    arms = step6.registered_arms()
    assert len(arms) == 22
    assert sum(30 if a['variant'] == 'P' else 45 for a in arms) == 900
    assert sum(30 if a['variant'] == 'P' else 45 for a in arms
               if a['alpha'] == .1 and a['n_train'] == 512) == 240
    assert len(step6.registered_arms(True)) == 30


def test_prefix_before_exclusion_and_capped_labels_retained():
    data = frame()
    X, z, ids = step6.training_rows(data, 128, 'z')
    assert len(z) == 127 and ids[0] == 'D_train_0000' and ids[-1] == 'D_train_0127'
    assert z[0] == 3 and 'D_train_0002' in ids
    for output in ('recovery', 'productivity'):
        X, y, ids = step6.training_rows(data, 128, output)
        assert len(y) == 126 and ids[0] == 'D_train_0002'
        assert 'D_train_0128' not in ids and 'D_train_0000' not in ids
        assert np.max(y) < 999


def test_gp_has_registered_ard_standardization_noise_and_restarts():
    gp = step6.make_gp()
    assert gp.normalize_y and gp.n_restarts_optimizer == 10 and gp.random_state == 0
    assert gp.kernel.k1.k2.nu == 2.5
    assert len(gp.kernel.k1.k2.length_scale) == 8
    assert gp.kernel.k2.noise_level_bounds[0] == 1e-6


@pytest.mark.parametrize('alpha,rank', [(.1, 232), (.05, 245)])
def test_calibration_uses_exact_order_statistic_not_interpolation(alpha, rank):
    q, actual = step6.conformal_quantile(np.arange(256)[::-1], alpha)
    assert actual == rank and q == rank - 1
    negative, _ = step6.conformal_quantile(-np.arange(1, 257), alpha)
    assert negative < 0


def test_small_calibration_unattainable_rank_is_infinite():
    assert step6.conformal_quantile([1., 2.], .05) == (float('inf'), 3)


def test_cp_gate_is_exact_one_sided_upper_not_two_sided_upper():
    lower, upper, one_sided = step6.cp_bounds(220, 256)
    assert lower < 220/256 < one_sided < upper
    # Inverting the binomial tail gives alpha=.05 at the one-sided endpoint.
    assert binom.cdf(220, 256, one_sided) == pytest.approx(.05)
    assert one_sided < .9  # this arm fails the registered .10 gate
    assert step6.cp_bounds(256, 256)[2] == 1
    assert step6.cp_bounds(0, 256)[0] == 0


def test_fixed_band_boundary_count_and_single_widening():
    values = np.r_[np.full(29, 1.5), np.full(10, 1.25), 2.01]
    mask, info = step6.fixed_band(values)
    assert info['widened'] and info['n_initial'] == 29 and info['n'] == 39
    assert not mask[-1]
    _, exact = step6.fixed_band(np.full(30, 2.))
    assert not exact['widened'] and exact['lower'] == 1.5
    _, sparse = step6.fixed_band([1.3, 2.5])
    assert sparse['widened'] and sparse['below_30'] and sparse['lower'] == 1.25


class FakeGP:
    def predict(self, X, return_std=False):
        mean = np.full(len(X), 1.)
        return (mean, np.full(len(X), .5)) if return_std else mean


def test_gp_local_and_gbm_cqr_have_different_registered_centers():
    gp = object.__new__(step6.Surrogates)
    gp.family = 'GP'
    gp.models = {'z': FakeGP()}
    X = np.zeros((2, 8))
    np.testing.assert_array_equal(gp.bound(X, 'L', .1, 2.), [2., 2.])
    np.testing.assert_array_equal(gp.bound(X, 'G', .1, 2.), [3., 3.])
    np.testing.assert_array_equal(gp.bound(X, 'P', .1, 2.), [1., 1.])
    gbm = object.__new__(step6.Surrogates)
    gbm.family = 'GBM'
    gbm.models = {'z': FakeGP()}
    gbm.quantiles = {.1: SimpleNamespace(predict=lambda X: np.full(len(X), 1.7))}
    np.testing.assert_array_equal(gbm.bound(X, 'L', .1, -.2), [1.5, 1.5])


def driver():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_step6b.py'
    spec = importlib.util.spec_from_file_location('step6b_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_refuses_new_fit_after_held_out_inspection(monkeypatch):
    d = driver()
    with pytest.raises(RuntimeError, match='No new fit'):
        d.train({'phases': {}, 'held_out_inspected': True}, 1)


def test_frozen_model_hash_detects_change(tmp_path, monkeypatch):
    d = driver()
    monkeypatch.setattr(d, 'WORK', tmp_path)
    (tmp_path/'models').mkdir()
    model = tmp_path/'models'/'model.joblib'
    model.write_bytes(b'frozen')
    state = {'phases': {'train': True}, 'fit_sha256': {model.name: step6.file_hash(model)}}
    d.verify_fit_freeze(state)
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        d.verify_fit_freeze(state)


def test_calibration_locks_fitting_before_first_held_out_label_read(monkeypatch):
    d = driver()
    state = {'phases': {'train': True}}
    saved = []
    monkeypatch.setattr(d, 'verify_fit_freeze', lambda state: None)
    monkeypatch.setattr(d, 'save', lambda state: saved.append(state.copy()))

    def read(name):
        assert name == 'D_cal' and state['held_out_inspected']
        assert saved[-1]['held_out_inspected']
        raise ValueError('synthetic read interruption')

    monkeypatch.setattr(d, 'read_set', read)
    with pytest.raises(ValueError, match='synthetic read interruption'):
        d.calibrate(state)
