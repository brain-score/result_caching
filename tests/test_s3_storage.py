"""S3-backed xarray storage.

Containers are ephemeral, so a local cache is worthless to them — the point of
caching activations is that an entry outlives the container that wrote it.

Tested against an in-memory fake rather than real S3: these assert the
storage protocol (what gets written, in what order, and what counts as a miss),
none of which needs a network.
"""
import json
import os

import numpy as np
import pytest
import xarray as xr

import result_caching
from result_caching import _MANIFEST_SCHEMA_VERSION, s3_enabled, store_xarray_s3


class FakeS3:
    """Minimal in-memory stand-in for the boto3 S3 client surface used here."""

    def __init__(self):
        self.objects = {}
        self.upload_calls = []
        self.download_calls = []

    def upload_file(self, local_path, bucket, key):
        with open(local_path, 'rb') as f:
            self.objects[(bucket, key)] = f.read()
        self.upload_calls.append(key)

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body
        self.upload_calls.append(Key)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(f'NoSuchKey {Key}')
        import io
        return {'Body': io.BytesIO(self.objects[(Bucket, Key)])}

    def download_file(self, bucket, key, local_path):
        if (bucket, key) not in self.objects:
            raise KeyError(f'NoSuchKey {key}')
        with open(local_path, 'wb') as f:
            f.write(self.objects[(bucket, key)])
        self.download_calls.append(key)


@pytest.fixture
def s3(tmp_path, monkeypatch):
    monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path))
    monkeypatch.setenv('RESULTCACHING_S3_BUCKET', 'test-bucket')
    monkeypatch.setenv('RESULTCACHING_S3_PREFIX', 'activations')
    monkeypatch.setenv('RESULTCACHING_S3_EPOCH', '1')
    monkeypatch.delenv('RESULTCACHING_DISABLE', raising=False)
    fake = FakeS3()
    monkeypatch.setattr(result_caching, '_s3_client', lambda: fake)
    return fake


def _assembly():
    return xr.DataArray(
        np.arange(6, dtype='float32').reshape(1, 2, 3),
        dims=['layer', 'presentation', 'neuroid'],
        coords={'layer': ['fc'], 'presentation': ['a', 'b'], 'neuroid': [0, 1, 2]})


def _counting_fn():
    calls = {'n': 0}

    @store_xarray_s3(identifier_ignore=[], combine_fields={'layers': 'layer'})
    def compute(identifier, layers):
        calls['n'] += 1
        return _assembly()

    return compute, calls


class TestOptionalDependency:
    def test_missing_boto3_degrades_to_no_cache(self, tmp_path, monkeypatch):
        """boto3 is an extra; absent, caching is off rather than broken."""
        monkeypatch.setenv('RESULTCACHING_HOME', str(tmp_path))
        monkeypatch.setenv('RESULTCACHING_S3_BUCKET', 'test-bucket')
        monkeypatch.delenv('RESULTCACHING_DISABLE', raising=False)
        monkeypatch.setattr(result_caching, '_s3_client', lambda: None)
        compute, calls = _counting_fn()
        assert np.asarray(compute(identifier='alexnet', layers=['fc'])).shape == (1, 2, 3)
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 2, "no client => always a miss, never an error"

    def test_real_client_helper_handles_absent_boto3(self, monkeypatch):
        """_s3_client itself must not raise when boto3 is not installed."""
        import builtins
        real_import = builtins.__import__

        def no_boto3(name, *args, **kwargs):
            if name == 'boto3':
                raise ImportError('No module named boto3')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', no_boto3)
        assert result_caching._s3_client() is None

    def test_s3_enabled_reflects_the_bucket_var(self, monkeypatch):
        monkeypatch.delenv('RESULTCACHING_S3_BUCKET', raising=False)
        assert s3_enabled() is False
        monkeypatch.setenv('RESULTCACHING_S3_BUCKET', 'b')
        assert s3_enabled() is True


class TestRoundTrip:
    def test_second_call_hits_s3(self, s3):
        compute, calls = _counting_fn()
        first = compute(identifier='alexnet', layers=['fc'])
        second = compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 1, "second call must come from S3"
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        assert s3.download_calls, "expected a download on the second call"

    def test_key_layout_includes_the_epoch(self, s3):
        compute, _ = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        data_keys = [k for k in s3.upload_calls if k.endswith('.nc')]
        assert len(data_keys) == 1
        assert data_keys[0].startswith('activations/epoch=1/'), data_keys[0]

    def test_bumping_the_epoch_invalidates_everything(self, s3, monkeypatch):
        compute, calls = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 1
        monkeypatch.setenv('RESULTCACHING_S3_EPOCH', '2')
        compute, calls2 = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        assert calls2['n'] == 1, "a new epoch must not read the old one"

    def test_staging_file_is_cleaned_up(self, s3, tmp_path):
        compute, _ = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        leftovers = list((tmp_path / '_s3_staging').rglob('*.nc')) if (tmp_path / '_s3_staging').exists() else []
        assert leftovers == [], f"staging files left behind: {leftovers}"


class TestWriteOrderGivesAtomicity:
    def test_data_is_written_before_the_manifest(self, s3):
        """A read requires the manifest, so this ordering makes an interrupted
        write indistinguishable from a miss."""
        compute, _ = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        nc_index = next(i for i, k in enumerate(s3.upload_calls) if k.endswith('.nc'))
        manifest_index = next(i for i, k in enumerate(s3.upload_calls) if k.endswith('.manifest.json'))
        assert nc_index < manifest_index

    def test_data_without_manifest_reads_as_a_miss(self, s3):
        compute, calls = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        for bucket, key in list(s3.objects):
            if key.endswith('.manifest.json'):
                del s3.objects[(bucket, key)]
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 2, "an interrupted write must be recomputed"


class TestUnusableEntryIsAMiss:
    def test_future_schema_version_is_skipped(self, s3):
        compute, calls = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        for (bucket, key), body in list(s3.objects.items()):
            if key.endswith('.manifest.json'):
                manifest = json.loads(body)
                manifest['schema_version'] = _MANIFEST_SCHEMA_VERSION + 99
                s3.objects[(bucket, key)] = json.dumps(manifest).encode()
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 2

    def test_manifest_records_the_writing_environment(self, s3):
        compute, _ = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        body = next(b for (_, k), b in s3.objects.items() if k.endswith('.manifest.json'))
        manifest = json.loads(body)
        assert manifest['schema_version'] == _MANIFEST_SCHEMA_VERSION
        assert {'xarray', 'pandas', 'numpy'} <= set(manifest['packages'])
        assert manifest['bytes'] > 0
        assert manifest['epoch'] == '1'


class TestFailuresNeverPropagate:
    def test_upload_failure_does_not_fail_the_call(self, s3):
        """Caching is an optimisation; losing it must not lose the result."""
        def boom(*args, **kwargs):
            raise RuntimeError('S3 is down')
        s3.upload_file = boom
        compute, calls = _counting_fn()
        result = compute(identifier='alexnet', layers=['fc'])  # must not raise
        np.testing.assert_array_equal(np.asarray(result), np.asarray(_assembly()))

    def test_unreadable_manifest_is_a_miss(self, s3):
        compute, calls = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        for (bucket, key) in list(s3.objects):
            if key.endswith('.manifest.json'):
                s3.objects[(bucket, key)] = b'not json'
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 2


class TestSizeCap:
    def test_oversized_results_are_not_cached(self, s3, monkeypatch):
        """The activation-size tail is most of the storage bill for a fraction
        of the benefit, so there is a cap."""
        monkeypatch.setenv('RESULTCACHING_S3_MAX_GB', '0')  # cap everything out
        compute, calls = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        assert [k for k in s3.upload_calls if k.endswith('.nc')] == []
        compute(identifier='alexnet', layers=['fc'])
        assert calls['n'] == 2, "nothing was cached, so this recomputes"

    def test_cap_default_is_finite_and_generous(self, s3):
        compute, _ = _counting_fn()
        compute(identifier='alexnet', layers=['fc'])
        assert [k for k in s3.upload_calls if k.endswith('.nc')], \
            "a small assembly must be under the default cap"
