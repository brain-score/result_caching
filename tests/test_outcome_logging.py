"""One INFO line per cached call, so hit rate is measurable in production.

Every other message in result_caching is DEBUG, which production never emits, so
a completed scoring run carried no evidence of whether the cache did anything.
"""
import logging

import numpy as np
import pytest
import xarray as xr

from result_caching import store, store_dict, store_xarray


@pytest.fixture
def cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path))
    monkeypatch.delenv('RESULTCACHING_DISABLE', raising=False)
    return tmp_path


def _outcomes(caplog):
    """(outcome, detail) pairs from the INFO lines, in order."""
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if 'RESULTCACHING outcome=' not in message:
            continue
        rest = message.split('RESULTCACHING outcome=', 1)[1]
        outcome = rest.split()[0]
        detail = rest.split('key=', 1)[0].strip()
        out.append((outcome, detail))
    return out


def _assembly(layers):
    values = np.arange(6 * len(layers), dtype='float32').reshape(len(layers), 2, 3)
    return xr.DataArray(values, dims=['layer', 'presentation', 'neuroid'],
                        coords={'layer': list(layers), 'presentation': ['a', 'b'],
                                'neuroid': [0, 1, 2]})


class TestPlainStore:
    def test_miss_then_hit(self, cache_home, caplog):
        @store()
        def compute(x):
            return x * 2

        with caplog.at_level(logging.INFO):
            assert compute(21) == 42
            assert compute(21) == 42
        assert [o for o, _ in _outcomes(caplog)] == ['miss', 'hit']

    def test_disabled_is_distinguished_from_a_miss(self, cache_home, monkeypatch, caplog):
        """Otherwise a run with caching off looks like a 100% miss rate rather
        than like a run that never consulted the cache."""
        monkeypatch.setenv('RESULTCACHING_DISABLE', '1')

        @store()
        def compute(x):
            return x * 2

        with caplog.at_level(logging.INFO):
            compute(21)
            compute(21)
        assert [o for o, _ in _outcomes(caplog)] == ['disabled', 'disabled']


class TestCombineFields:
    """`partial` is the activation cache's normal case: one entry accumulates
    layers across calls, so reporting it as hit or miss would misstate it."""

    def _counting_fn(self):
        calls = {'n': 0}

        # identifier_ignore must exclude the combine field or `layers` stays in
        # the key and a superset call is a different entry -- this is exactly how
        # brainscore_vision decorates _from_paths_stored.
        @store_xarray(identifier_ignore=['layers'], combine_fields={'layers': 'layer'})
        def compute(identifier, layers):
            calls['n'] += 1
            return _assembly(layers)

        return compute, calls

    def test_miss_then_partial_then_hit(self, cache_home, caplog):
        compute, calls = self._counting_fn()
        with caplog.at_level(logging.INFO):
            compute('m', layers=['fc'])              # nothing stored -> miss
            compute('m', layers=['fc', 'conv'])      # fc reused, conv computed -> partial
            compute('m', layers=['fc', 'conv'])      # both stored -> hit
        assert [o for o, _ in _outcomes(caplog)] == ['miss', 'partial', 'hit']
        assert calls['n'] == 2, "the partial call must compute only the missing layer"

    def test_partial_names_what_was_missing(self, cache_home, caplog):
        compute, _ = self._counting_fn()
        compute('m', layers=['fc'])
        with caplog.at_level(logging.INFO):
            compute('m', layers=['fc', 'conv'])
        outcome, detail = _outcomes(caplog)[0]
        assert outcome == 'partial'
        assert 'conv' in detail and 'fc' not in detail, \
            f"detail should name only the missing layer, got {detail!r}"

    def test_a_subset_of_stored_layers_is_a_hit(self, cache_home, caplog):
        compute, calls = self._counting_fn()
        compute('m', layers=['fc', 'conv'])
        before = calls['n']
        with caplog.at_level(logging.INFO):
            compute('m', layers=['fc'])
        assert [o for o, _ in _outcomes(caplog)] == ['hit']
        assert calls['n'] == before, "a subset must not recompute"


class TestDictStore:
    def test_miss_then_partial(self, cache_home, caplog):
        calls = {'n': 0}

        @store_dict(dict_key='keys', identifier_ignore=['keys'])
        def compute(identifier, keys):
            calls['n'] += 1
            return {k: k * 2 for k in keys}

        with caplog.at_level(logging.INFO):
            compute('d', keys=[1])
            compute('d', keys=[1, 2])
        assert [o for o, _ in _outcomes(caplog)] == ['miss', 'partial']
        assert calls['n'] == 2


class TestOneLinePerCall:
    def test_exactly_one_outcome_per_call(self, cache_home, caplog):
        """Log volume is the reason these are not DEBUG-promoted wholesale: at
        ~125 jobs a run, one line per call is greppable and more is noise."""
        @store()
        def compute(x):
            return x

        with caplog.at_level(logging.INFO):
            for i in range(10):
                compute(i)
        assert len(_outcomes(caplog)) == 10
