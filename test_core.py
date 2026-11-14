"""Quick correctness checks.  Run:  python tests/test_core.py"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import tep_fd as T  # noqa: E402


def _data(n, seed):
    rng = np.random.default_rng(seed)
    A = np.random.default_rng(99).normal(size=(10, 3))
    return rng.normal(size=(n, 3)) @ A.T + 0.3 * rng.normal(size=(n, 10))


def test_contributions_sum_to_statistics():
    X = _data(4000, 0)
    m = T.PCAMonitor(n_components=3).fit(X[:3000], X[3000:])
    t2, spe = m.stats(X[3000:])
    c_t2, c_spe = m.contributions(X[3000:])
    assert np.allclose(c_t2.sum(axis=1), t2)
    assert np.allclose(c_spe.sum(axis=1), spe)


def test_false_alarm_rate_close_to_alpha():
    X_fit, X_val, X_test = _data(10000, 1), _data(5000, 2), _data(20000, 3)
    m = T.PCAMonitor(n_components=3, alpha=0.99).fit(X_fit, X_val)
    a = m.alarms(X_test)
    for key in ("alarm_t2", "alarm_spe"):
        rate = a[key].mean() * 100
        assert 0.3 <= rate <= 2.0, f"{key} false alarm rate {rate:.2f}% is far from 1%"


def test_mean_shift_is_detected():
    X_fit, X_val = _data(10000, 4), _data(5000, 5)
    m = T.PCAMonitor(n_components=3).fit(X_fit, X_val)
    X_bad = _data(2000, 6)
    X_bad[:, 0] += 8 * X_fit[:, 0].std()
    assert m.alarms(X_bad)["alarm_pca"].mean() > 0.9


def test_first_sustained():
    a = np.array([0, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert T.first_sustained(a, 0, 3) == 3
    assert T.first_sustained(a, 4, 3) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all tests passed")
