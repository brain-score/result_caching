"""netCDF + manifest storage for _XarrayStorage.

The default pickle format cannot be read back across a pandas/xarray upgrade,
and `pd.read_pickle` raises rather than missing cleanly, so an environment bump
turns a warm cache into a failing one. These tests pin the two properties that
make the netCDF format safe for a shared, long-lived cache:

1. Switching format never corrupts or collides with an existing cache.
2. An entry that cannot be used in this environment reads as a MISS, never as
   an error -- so drift costs one recomputation, which is what it costs today.
"""
import json
import os

import numpy as np
import pytest
import xarray as xr

from result_caching import _MANIFEST_SCHEMA_VERSION, _manifest_path, store_xarray, use_netcdf_format


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path))
    monkeypatch.delenv('RESULTCACHING_DISABLE', raising=False)
    return tmp_path


@pytest.fixture
def netcdf(monkeypatch):
    monkeypatch.setenv('RESULTCACHING_FORMAT', 'netcdf')


def _assembly(layers=('fc',)):
    values = np.arange(6 * len(layers), dtype='float32').reshape(len(layers), 2, 3)
    return xr.DataArray(
        values, dims=['layer', 'presentation', 'neuroid'],
        coords={'layer': list(layers),
                'presentation': ['a', 'b'],
                'neuroid': [0, 1, 2]})


def _make_counting_fn():
    calls = {'n': 0}

    @store_xarray(identifier_ignore=[], combine_fields={'layers': 'layer'})
    def compute(identifier, layers):
        calls['n'] += 1
        return _assembly(layers)

    return compute, calls


class TestFormatSelection:
    def test_pickle_is_the_default(self, monkeypatch):
        monkeypatch.delenv('RESULTCACHING_FORMAT', raising=False)
        assert use_netcdf_format() is False

    @pytest.mark.parametrize('value,expected', [
        ('netcdf', True), ('nc', True), ('NetCDF', True),
        ('pickle', False), ('', False),
    ])
    def test_flag_parsing(self, monkeypatch, value, expected):
        monkeypatch.setenv('RESULTCACHING_FORMAT', value)
        assert use_netcdf_format() is expected

    def test_netcdf_writes_nc_and_a_manifest(self, cache_dir, netcdf):
        compute, calls = _make_counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        written = [p for p in cache_dir.rglob('*') if p.is_file()]
        assert any(p.suffix == '.nc' for p in written), [p.name for p in written]
        assert any(p.name.endswith('.manifest.json') for p in written)
        assert not any(p.suffix == '.pkl' for p in written)

    def test_switching_format_misses_rather_than_breaking(self, cache_dir, monkeypatch):
        """A cache written as pickle must not be misread as netCDF."""
        monkeypatch.delenv('RESULTCACHING_FORMAT', raising=False)
        compute, calls = _make_counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 1
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 1, "pickle cache should hit"

        monkeypatch.setenv('RESULTCACHING_FORMAT', 'netcdf')
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 2, "different format => clean miss, not an error"


class TestRoundTrip:
    def test_values_survive_a_round_trip(self, cache_dir, netcdf):
        compute, calls = _make_counting_fn()
        first = compute(identifier='alexnet', layers=['fc'])
        second = compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 1, "second call must hit the cache"
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_dtype_is_preserved(self, cache_dir, netcdf):
        compute, _ = _make_counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        loaded = compute(identifier='alexnet', layers=['fc'])
        assert np.asarray(loaded).dtype == np.dtype('float32')

    def test_manifest_records_the_writing_environment(self, cache_dir, netcdf):
        compute, _ = _make_counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        manifest_file = next(p for p in cache_dir.rglob('*.manifest.json'))
        manifest = json.loads(manifest_file.read_text())
        assert manifest['schema_version'] == _MANIFEST_SCHEMA_VERSION
        assert {'xarray', 'pandas', 'numpy'} <= set(manifest['packages'])
        assert manifest['bytes'] > 0


class TestUnusableEntryIsAMiss:
    """The property that makes environment drift free instead of fatal."""

    def _write_once(self, cache_dir, netcdf_env):
        compute, calls = _make_counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        nc = next(p for p in cache_dir.rglob('*.nc'))
        return compute, calls, nc

    def test_future_schema_version_is_skipped(self, cache_dir, netcdf):
        compute, calls, nc = self._write_once(cache_dir, netcdf)
        manifest_file = nc.parent / (nc.name + '.manifest.json')
        manifest = json.loads(manifest_file.read_text())
        manifest['schema_version'] = _MANIFEST_SCHEMA_VERSION + 99
        manifest_file.write_text(json.dumps(manifest))
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 2, "an entry from a newer writer must be recomputed, not read"

    def test_missing_manifest_is_skipped(self, cache_dir, netcdf):
        compute, calls, nc = self._write_once(cache_dir, netcdf)
        (nc.parent / (nc.name + '.manifest.json')).unlink()
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 2

    def test_size_mismatch_is_skipped(self, cache_dir, netcdf):
        """Catches truncation or external mangling."""
        compute, calls, nc = self._write_once(cache_dir, netcdf)
        with open(nc, 'ab') as f:
            f.write(b'trailing garbage')
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 2

    def test_unreadable_file_is_a_miss_not_an_exception(self, cache_dir, netcdf):
        """The catch-all: whatever makes a file unopenable, recompute."""
        compute, calls, nc = self._write_once(cache_dir, netcdf)
        manifest_file = nc.parent / (nc.name + '.manifest.json')
        nc.write_bytes(b'not netcdf at all')
        # keep the manifest consistent with the new size so the size check
        # passes and the open() guard is what has to catch this
        manifest = json.loads(manifest_file.read_text())
        manifest['bytes'] = os.path.getsize(nc)
        manifest_file.write_text(json.dumps(manifest))

        result = compute(identifier='alexnet', layers=['fc'])  # must not raise
        assert calls['n'] == 2
        np.testing.assert_array_equal(np.asarray(result), np.asarray(_assembly(['fc'])))


class TestPickleFormatUnaffected:
    """Local developers keep the behaviour they have today."""

    def test_pickle_path_and_hit(self, cache_dir, monkeypatch):
        monkeypatch.delenv('RESULTCACHING_FORMAT', raising=False)
        compute, calls = _make_counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 1
        assert any(p.suffix == '.pkl' for p in cache_dir.rglob('*') if p.is_file())
        assert not any(p.name.endswith('.manifest.json') for p in cache_dir.rglob('*'))


class TestRealAssemblySubclass:
    """Regression: the writer used to rebuild with `type(result)(...)`.

    brainio's DataAssembly subclasses re-derive a MultiIndex from same-dim
    coords in __init__, so that reconstruction immediately undid the flattening
    and `to_netcdf` raised NotImplementedError on every real assembly. Tests
    using a plain xr.DataArray could not see it, because DataArray does not
    re-derive anything.

    Skipped where brainscore_core is unavailable; result_caching does not depend
    on it.
    """

    @pytest.fixture
    def assembly_class(self):
        mod = pytest.importorskip(
            'brainscore_core.supported_data_standards.brainio.assemblies')
        return mod.NeuroidAssembly

    def _assembly(self, cls, n_stim=3, n_neuroid=8):
        return cls(
            np.arange(n_stim * n_neuroid, dtype='float32').reshape(n_stim, n_neuroid),
            coords={'stimulus_path': ('presentation', [f'/i/{i}.png' for i in range(n_stim)]),
                    'stimulus_id': ('presentation', [f's{i}' for i in range(n_stim)]),
                    'neuroid_id': ('neuroid', list(range(n_neuroid))),
                    'layer': ('neuroid', ['fc'] * n_neuroid),
                    'channel': ('neuroid', list(range(n_neuroid)))},
            dims=['presentation', 'neuroid'])

    def test_multiindex_assembly_round_trips(self, cache_dir, netcdf, assembly_class):
        source = self._assembly(assembly_class)
        assert len(source.indexes) == 2, "fixture should have MultiIndex on both dims"
        calls = {'n': 0}

        @store_xarray(identifier_ignore=[], combine_fields={'layers': 'layer'})
        def compute(identifier, layers):
            calls['n'] += 1
            return source

        first = compute(identifier='alexnet', layers=['fc'])
        second = compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 1, "second call must hit the cache"
        np.testing.assert_array_equal(
            np.asarray(first.transpose('presentation', 'neuroid')),
            np.asarray(second.transpose('presentation', 'neuroid')))

    def test_assembly_class_is_restored(self, cache_dir, netcdf, assembly_class):
        """Written as a plain DataArray; the subclass must come back, or callers
        lose the assembly API they stored."""
        source = self._assembly(assembly_class)

        @store_xarray(identifier_ignore=[], combine_fields={'layers': 'layer'})
        def compute(identifier, layers):
            return source

        compute(identifier='alexnet', layers=['fc'])
        loaded = compute(identifier='alexnet', layers=['fc'])
        assert type(loaded) is assembly_class, type(loaded)

    def test_multiindex_is_restored(self, cache_dir, netcdf, assembly_class):
        source = self._assembly(assembly_class)

        @store_xarray(identifier_ignore=[], combine_fields={'layers': 'layer'})
        def compute(identifier, layers):
            return source

        compute(identifier='alexnet', layers=['fc'])
        loaded = compute(identifier='alexnet', layers=['fc'])
        assert {n: list(loaded.indexes[n].names) for n in loaded.indexes} == \
               {n: list(source.indexes[n].names) for n in source.indexes}

    def test_manifest_records_the_class(self, cache_dir, netcdf, assembly_class):
        source = self._assembly(assembly_class)

        @store_xarray(identifier_ignore=[], combine_fields={'layers': 'layer'})
        def compute(identifier, layers):
            return source

        compute(identifier='alexnet', layers=['fc'])
        manifest = json.loads(next(cache_dir.rglob('*.manifest.json')).read_text())
        assert manifest['class']['name'] == 'NeuroidAssembly'

    def test_unresolvable_class_still_loads(self, cache_dir, netcdf, assembly_class):
        """If the recorded class cannot be imported, fall back to a DataArray
        rather than failing the read."""
        source = self._assembly(assembly_class)

        @store_xarray(identifier_ignore=[], combine_fields={'layers': 'layer'})
        def compute(identifier, layers):
            return source

        compute(identifier='alexnet', layers=['fc'])
        manifest_file = next(cache_dir.rglob('*.manifest.json'))
        manifest = json.loads(manifest_file.read_text())
        manifest['class'] = {'module': 'no.such.module', 'name': 'Gone'}
        manifest_file.write_text(json.dumps(manifest))
        loaded = compute(identifier='alexnet', layers=['fc'])   # must not raise
        assert loaded is not None
