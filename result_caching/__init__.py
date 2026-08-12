from collections import defaultdict, OrderedDict

import inspect
import itertools
import json
import logging
import numpy as np
import pandas as pd
import os
import pickle
import xarray as xr
from functools import wraps
from typing import Union


def get_function_identifier(function, call_args):
    module = [function.__module__, function.__name__]
    if 'self' in call_args:
        object = call_args['self']
        class_name = object.__class__.__name__
        if 'object at' in str(object):
            object = class_name
        else:
            object = f"{class_name}({str(object)})"
        module.insert(1, object)
        del call_args['self']
    module = '.'.join(module)
    strip_slashes = lambda x: str(x).replace('/', '_')
    params = ','.join(f'{key}={strip_slashes(value)}' for key, value in call_args.items())
    if params:
        function_identifier = os.path.join(module, params)
    else:
        function_identifier = module
    return function_identifier


def is_enabled(function_identifier):
    disable = os.getenv('RESULTCACHING_DISABLE', '0')
    return not _match_identifier(function_identifier, disable)


def cached_only(function_identifier):
    cachedonly = os.getenv('RESULTCACHING_CACHEDONLY', '0')
    return _match_identifier(function_identifier, cachedonly)


def _match_identifier(function_identifier, match_value):
    if match_value == '1':
        return True
    if match_value == '':
        return False
    disabled_modules = match_value.split(',')
    return any(function_identifier.startswith(disabled_module) for disabled_module in disabled_modules)


def log_outcome(logger, outcome: str, function_identifier: str, detail: str = '') -> None:
    """Emit one INFO line recording what the cache did for this call.

    Every other message in this module is DEBUG, which production never emits,
    so a completed run carried no evidence of whether the cache did anything --
    hit rate was unmeasurable after the fact, which defeats the point of having
    a shared cache at all.

    One line per cached call, at INFO, greppable as ``RESULTCACHING outcome=``.

    Outcomes:
      ``hit``      served entirely from storage; the function did not run
      ``miss``     nothing stored; the function ran and the result was saved
      ``partial``  some fields loaded, the rest computed and merged in. Only
                   reachable with ``combine_fields``, and it is the *normal*
                   case for the activation cache, where one entry accumulates
                   layers across calls -- reporting it as a plain hit or miss
                   would misstate what happened.
      ``disabled`` caching is off for this identifier; nothing read or written
    """
    logger.info("RESULTCACHING outcome=%s%s key=%s", outcome, detail, function_identifier)


class NotCachedError(Exception):
    pass


class _Storage(object):
    def __init__(self, identifier_ignore=()):
        """
        :param [str] identifier_ignore: function parameters to ignore when building the unique function identifier.
            Different versions of the same parameter will result in the same identifier when ignored.
            Useful when the results do not depend on certain parameters.
        """
        self.identifier_ignore = identifier_ignore
        self._logger = logging.getLogger(_fullname(self))

    def __call__(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            call_args = self.getcallargs(function, *args, **kwargs)
            function_identifier = self.get_function_identifier(function, call_args)
            enabled = is_enabled(function_identifier)
            if enabled and self.is_stored(function_identifier):
                self._logger.debug("Loading from storage: {}".format(function_identifier))
                log_outcome(self._logger, 'hit', function_identifier)
                return self.load(function_identifier)
            if cached_only(function_identifier):
                raise NotCachedError(f"No result stored for '{function_identifier}'")
            self._logger.debug("Running function: {}".format(function_identifier))
            log_outcome(self._logger, 'miss' if enabled else 'disabled', function_identifier)
            result = function(*args, **kwargs)
            if enabled:
                self._logger.debug("Saving to storage: {}".format(function_identifier))
                self.save(result, function_identifier)
            return result

        return wrapper

    def getcallargs(self, function, *args, **kwargs):
        call_args = inspect.getcallargs(function, *args, **kwargs)
        argspec = inspect.getfullargspec(function)
        argspec = argspec.args + \
                  ([argspec.varargs] if argspec.varargs else []) + ([argspec.varkw] if argspec.varkw else [])
        sorting = {arg: i for i, arg in enumerate(argspec)}
        return OrderedDict(sorted(call_args.items(), key=lambda pair: sorting[pair[0]]))

    def get_function_identifier(self, function, call_args):
        call_args = {key: value for key, value in call_args.items() if key not in self.identifier_ignore}
        return get_function_identifier(function, call_args)

    def is_stored(self, function_identifier):
        raise NotImplementedError()

    def load(self, function_identifier):
        raise NotImplementedError()

    def save(self, result, function_identifier):
        raise NotImplementedError()


class _DiskStorage(_Storage):
    def __init__(self, identifier_ignore=()):
        super().__init__(identifier_ignore=identifier_ignore)
        self._storage_directory = os.path.expanduser(os.getenv('RESULTCACHING_HOME', '~/.result_caching'))

    def storage_path(self, function_identifier):
        return os.path.join(self._storage_directory, function_identifier + '.pkl')

    def save(self, result, function_identifier):
        path = self.storage_path(function_identifier)
        path_dir = os.path.dirname(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir, exist_ok=True)
        savepath_part = path + '.filepart'
        self.save_file(result, savepath_part)
        os.rename(savepath_part, path)

    def save_file(self, result, savepath_part):
        with open(savepath_part, 'wb') as f:
            pickle.dump({'data': result}, f, protocol=-1)  # highest protocol

    def is_stored(self, function_identifier):
        storage_path = self.storage_path(function_identifier)
        return os.path.isfile(storage_path)

    def load(self, function_identifier):
        path = self.storage_path(function_identifier)
        assert os.path.isfile(path)
        return self.load_file(path)

    def load_file(self, path):
        with open(path, 'rb') as f:
            return pd.read_pickle(f)['data']



def _netcdf_walk_coords(assembly):
    """Yield (name, (dims, values)) for every coord, expanding MultiIndex levels."""
    for name, values in assembly.coords.items():
        is_index = name in assembly.dims
        if is_index and values.variable.level_names:
            for level in values.variable.level_names:
                level_values = assembly.coords[level]
                yield level, (level_values.dims, level_values.values)
        else:
            yield name, (values.dims, values.values)


def _netcdf_save(result, savepath):
    """Write an xarray result to netCDF with MultiIndex coords flattened.

    The flattening is the point: MultiIndex round-tripping is the least stable
    part of the xarray/pandas boundary across versions, so it is resolved to
    plain coords on the way out and rebuilt by the caller on the way in.
    """
    result_coords = [coord for coord, values in _netcdf_walk_coords(result)]
    # `list(...)`: recent xarray rejects a KeysView here, which is why the
    # pre-existing _NetcdfStorage writer raised TypeError on any assembly.
    indexes = list(result.indexes.keys())
    if indexes:
        result = result.reset_index(indexes)
    # the reset suffixes single-index coordinates with '_'
    coords = {}
    for coord, values in _netcdf_walk_coords(result):
        if coord not in result_coords:
            assert coord.endswith('_') and coord[:-1] in result_coords
            coord = coord[:-1]
        coords[coord] = values
    # Rebuild as a plain DataArray, NOT type(result): brainio's DataAssembly
    # subclasses re-derive a MultiIndex from same-dim coords in __init__, which
    # would undo the flattening we just did and make to_netcdf raise
    # NotImplementedError. The subclass is recorded in the manifest and restored
    # on load instead.
    result = xr.DataArray(result.values, coords=coords, dims=result.dims)
    result.to_netcdf(savepath)


def _netcdf_load(path, target_class=None):
    """Read a netCDF cache entry, restoring the original assembly class.

    Written as a plain DataArray with flat coords; the recorded subclass
    re-derives its MultiIndex from those coords on construction, so callers get
    back the type they stored.
    """
    loaded = xr.open_dataarray(path)
    if target_class is None:
        return loaded
    loaded.load()  # detach from the file before it is deleted or reused
    return target_class(loaded)


def _describe_class(result):
    cls = type(result)
    return {'module': cls.__module__, 'name': cls.__qualname__}


def _resolve_class(described):
    """Import a class recorded by _describe_class, or None if unavailable."""
    if not described:
        return None
    try:
        import importlib
        module = importlib.import_module(described['module'])
        return getattr(module, described['name'])
    except Exception:
        logging.getLogger(__name__).debug(
            "Could not resolve cached assembly class %s", described, exc_info=True)
        return None


class _NetcdfStorage(_DiskStorage):
    def storage_path(self, function_identifier):
        return os.path.join(self._storage_directory, function_identifier + '.nc')

    def save_file(self, result, savepath_part):
        _netcdf_save(result, savepath_part)

    def load_file(self, path):
        return _netcdf_load(path)

    @classmethod
    def walk_coords(cls, assembly):
        """
        walks through coords and all levels, just like the `__repr__` function, yielding `(name, dims, values)`.
        """
        coords = {}

        for name, values in assembly.coords.items():
            # partly borrowed from xarray.core.formatting#summarize_coord
            is_index = name in assembly.dims
            if is_index and values.variable.level_names:
                for level in values.variable.level_names:
                    level_values = assembly.coords[level]
                    yield level, (level_values.dims, level_values.values)
            else:
                yield name, (values.dims, values.values)
        return coords


class _DictStorage(_DiskStorage):
    """
    All fields in _combine_fields are combined into one file and loaded lazily
    """

    def __init__(self, dict_key: str, *args, **kwargs):
        """
        :param dict_key: the argument representing the dictionary key.
        """
        super().__init__(*args, **kwargs)
        self._dict_key = dict_key

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            call_args = self.getcallargs(function, *args, **kwargs)
            assert self._dict_key in call_args
            infile_call_args = {self._dict_key: call_args[self._dict_key]}
            function_identifier = self.get_function_identifier(function, call_args)
            stored_result, reduced_call_args = None, call_args
            if is_enabled(function_identifier) and self.is_stored(function_identifier):
                self._logger.debug(f"Loading from storage: {function_identifier}")
                stored_result = self.load(function_identifier)
                infile_missing_call_args = self.missing_call_args(infile_call_args, stored_result)
                if len(infile_missing_call_args) == 0:
                    # nothing else to run, but still need to filter
                    log_outcome(self._logger, 'hit', function_identifier)
                    result = stored_result
                    reduced_call_args = None
                else:
                    # need to run more args
                    non_variable_call_args = {key: value for key, value in call_args.items() if key != self._dict_key}
                    infile_missing_call_args = {self._dict_key: infile_missing_call_args}
                    reduced_call_args = {**non_variable_call_args, **infile_missing_call_args}
                    self._logger.debug(f"Computing missing: {reduced_call_args}")
                    log_outcome(self._logger, 'partial', function_identifier,
                                detail=f" missing={sorted(infile_missing_call_args[self._dict_key])}")
            if reduced_call_args:
                if cached_only(function_identifier):
                    raise NotCachedError(f"The following arguments for '{function_identifier}' "
                                         f"are not stored: {reduced_call_args}")
                # run function if some args are uncomputed
                self._logger.debug(f"Running function: {function_identifier}")
                if stored_result is None:  # nothing loaded => not a partial
                    log_outcome(self._logger,
                                'miss' if is_enabled(function_identifier) else 'disabled',
                                function_identifier)
                result = function(**reduced_call_args)
                if not self.callargs_present(result, {self._dict_key: reduced_call_args[self._dict_key]}):
                    raise ValueError("result does not contain requested keys")
                if stored_result is not None:
                    result = self.merge_results(stored_result, result)
                # only save if new results
                if is_enabled(function_identifier):
                    self._logger.debug("Saving to storage: {}".format(function_identifier))
                    self.save(result, function_identifier)
            assert self.callargs_present(result, infile_call_args)
            result = self.filter_callargs(result, infile_call_args)
            return result

        return wrapper

    def merge_results(self, stored_result, result):
        return {**stored_result, **result}

    def callargs_present(self, result, infile_call_args):
        # make sure coords are set equal to call_args
        return len(self.missing_call_args(infile_call_args, result)) == 0

    def missing_call_args(self, call_args, data):
        assert len(call_args) == 1 and list(call_args.keys())[0] == self._dict_key
        keys = list(call_args.values())[0]
        return [key for key in keys if key not in data]

    def filter_callargs(self, data, call_args):
        assert len(call_args) == 1 and list(call_args.keys())[0] == self._dict_key
        keys = list(call_args.values())[0]
        return type(data)((key, value) for key, value in data.items() if key in keys)


# --------------------------------------------------------------------------
# netCDF + manifest format for _XarrayStorage
#
# The default format is pickle, which cannot be read back across a pandas or
# xarray upgrade -- a cache written before an environment bump becomes so much
# dead weight, and `pd.read_pickle` raises rather than missing cleanly. That is
# tolerable for a scratch cache on one machine and not tolerable for a shared,
# long-lived one.
#
# Setting RESULTCACHING_FORMAT=netcdf switches to netCDF plus a JSON manifest.
# netCDF is self-describing and reads back across versions, and the manifest
# records what wrote the entry so an incompatible one can be recognised and
# skipped instead of raising.
#
# The two formats use different file extensions, so switching does not corrupt
# or collide with an existing cache -- it simply misses and recomputes.
# --------------------------------------------------------------------------

_S3_BUCKET_VAR = 'RESULTCACHING_S3_BUCKET'
_S3_PREFIX_VAR = 'RESULTCACHING_S3_PREFIX'
_S3_EPOCH_VAR = 'RESULTCACHING_S3_EPOCH'
_S3_MAX_GB_VAR = 'RESULTCACHING_S3_MAX_GB'

# Refuse to write anything larger than this. The activation-size distribution
# has a long tail (one vision model projects to 146 GB against a p90 of 31 GB),
# and that tail is most of the storage bill for a fraction of the benefit.
_S3_DEFAULT_MAX_GB = 50.0

def s3_enabled() -> bool:
    return bool(os.getenv(_S3_BUCKET_VAR, '').strip())


def _s3_client():
    """boto3 S3 client, or None when boto3 is not installed.

    Optional dependency: a missing boto3 must degrade to "no cache", never
    break an import or a run.
    """
    try:
        import boto3
    except ImportError:
        # Only worth flagging when someone actually asked for S3 caching.
        if s3_enabled():
            logging.getLogger(__name__).warning(
                "%s is set but boto3 is not installed; S3 caching is off. "
                "Install with `pip install result_caching[s3]`.", _S3_BUCKET_VAR)
        return None
    return boto3.client('s3')

# Bump when the on-disk layout changes in a way older readers cannot handle.
_MANIFEST_SCHEMA_VERSION = 1

_FORMAT_VAR = 'RESULTCACHING_FORMAT'


def use_netcdf_format() -> bool:
    """True if xarray results should be stored as netCDF + manifest."""
    return os.getenv(_FORMAT_VAR, 'pickle').strip().lower() in ('netcdf', 'nc')


def _manifest_path(storage_path):
    return storage_path + '.manifest.json'


def _write_manifest(storage_path, result):
    """Record what produced this entry, next to the entry itself."""
    import numpy
    import xarray
    manifest = {
        'schema_version': _MANIFEST_SCHEMA_VERSION,
        'packages': {'xarray': xarray.__version__, 'pandas': pd.__version__,
                     'numpy': numpy.__version__},
        'dtype': str(getattr(result, 'dtype', '')),
        'shape': list(getattr(result, 'shape', ())),
        'bytes': os.path.getsize(storage_path) if os.path.isfile(storage_path) else None,
        'class': _describe_class(result),
    }
    with open(_manifest_path(storage_path), 'w') as f:
        json.dump(manifest, f, sort_keys=True)


def _class_from_manifest(manifest_file):
    try:
        with open(manifest_file) as f:
            return _resolve_class(json.load(f).get('class'))
    except Exception:
        return None


def _manifest_is_usable(storage_path) -> bool:
    """Whether this entry can be trusted enough to attempt a read.

    Deliberately cheap: schema version and byte size only. Writes are atomic
    (temp file then rename) so a torn file is not a realistic failure mode, and
    checksumming a multi-gigabyte activation array on every read would cost far
    more than the extraction it saves. The manifest still records a size so a
    truncated or externally-mangled file is caught.
    """
    path = _manifest_path(storage_path)
    if not os.path.isfile(path):
        # Written by an older version, or the manifest was lost. Treat as
        # unusable rather than guessing -- recomputing is always safe.
        return False
    try:
        with open(path) as f:
            manifest = json.load(f)
    except Exception:
        return False
    if manifest.get('schema_version') != _MANIFEST_SCHEMA_VERSION:
        _logger = logging.getLogger(__name__)
        _logger.debug(f"Cache entry {storage_path} has schema_version "
                      f"{manifest.get('schema_version')}, expected {_MANIFEST_SCHEMA_VERSION}")
        return False
    expected_bytes = manifest.get('bytes')
    if expected_bytes is not None and os.path.getsize(storage_path) != expected_bytes:
        return False
    return True


class _XarrayStorage(_DiskStorage):
    """
    All fields in _combine_fields are combined into one file and loaded lazily
    """

    def __init__(self, combine_fields: Union[list, dict], sub_fields=False,
                 map_field_values=None, map_field_values_inverse=None,
                 *args, **kwargs):
        """
        :param combine_fields: fields to consider as primary keys together with the filename
            (i.e. fields not excluded by `identifier_ignore`).
        :param sub_fields: store the result right away (default, False) or only its sub-fields
        """
        super().__init__(*args, **kwargs)
        if not isinstance(combine_fields, dict):  # use identity mapping if list passed
            self._combine_fields = {field: field for field in combine_fields}
        else:
            self._combine_fields = combine_fields
        self._combine_fields_inverse = {value: key for key, value in self._combine_fields.items()}
        self._sub_fields = sub_fields
        if map_field_values:
            assert map_field_values_inverse
        self._map_field_values = map_field_values or (lambda key, value: value)
        self._map_field_values_inverse = map_field_values_inverse or (lambda key, value: value)

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            call_args = self.getcallargs(function, *args, **kwargs)
            infile_call_args = {self._combine_fields[key]: self._map_field_values(self._combine_fields[key], value)
                                for key, value in call_args.items()
                                if key in self._combine_fields}
            function_identifier = self.get_function_identifier(function, call_args)
            stored_result, reduced_call_args = None, call_args
            if is_enabled(function_identifier) and self.is_stored(function_identifier):
                self._logger.debug(f"Loading from storage: {function_identifier}")
                stored_result = self.load(function_identifier)
                missing_call_args = self.filter_coords(infile_call_args, stored_result) if not self._sub_fields \
                    else self.filter_coords(infile_call_args, getattr(stored_result, next(iter(vars(stored_result)))))
                if len(missing_call_args) == 0:
                    # nothing else to run, but still need to filter
                    log_outcome(self._logger, 'hit', function_identifier)
                    result = stored_result
                    reduced_call_args = None
                else:
                    # need to run more args
                    non_variable_call_args = {key: value for key, value in call_args.items()
                                              if key not in self._combine_fields}
                    missing_call_args = {self._combine_fields_inverse[key]: self._map_field_values_inverse(key, value)
                                         for key, value in missing_call_args.items()}
                    reduced_call_args = {**non_variable_call_args, **missing_call_args}
                    self._logger.debug(f"Computing missing: {reduced_call_args}")
                    # values, not keys: missing_call_args has been remapped to
                    # {outer_arg: values}, so sorted() would report the arg name.
                    _missing = sorted(str(value) for values in missing_call_args.values()
                                      for value in np.atleast_1d(values))
                    log_outcome(self._logger, 'partial', function_identifier,
                                detail=f" missing={_missing}")
            if reduced_call_args:
                if cached_only(function_identifier):
                    raise NotCachedError(f"The following arguments for '{function_identifier}' "
                                         f"are not stored: {reduced_call_args}")
                self._logger.debug(f"Running function: {function_identifier}")
                if stored_result is None:  # nothing loaded => not a partial
                    log_outcome(self._logger,
                                'miss' if is_enabled(function_identifier) else 'disabled',
                                function_identifier)
                # run function if some args are uncomputed
                result = function(**reduced_call_args)
                if stored_result is not None:
                    result = self.merge_results(stored_result, result)
                # only save if new results
                if is_enabled(function_identifier):
                    self._logger.debug("Saving to storage: {}".format(function_identifier))
                    self.save(result, function_identifier)
            self.ensure_callargs_present(result, infile_call_args)
            result = self.filter_callargs(result, infile_call_args)
            return result

        return wrapper

    def merge_results(self, stored_result, result):
        if not self._sub_fields:
            result = self._merge_data_arrays([stored_result, result])
        else:
            for field in vars(result):
                setattr(result, field,
                        self._merge_data_arrays([getattr(stored_result, field), getattr(result, field)]))
        return result

    def _merge_data_arrays(self, data_arrays):
        # https://stackoverflow.com/a/50125997/2225200
        merged = xr.merge([similarity.rename('z') for similarity in data_arrays])['z'].rename(None)
        # ensure same class
        return type(data_arrays[0])(merged)

    def ensure_callargs_present(self, result, infile_call_args):
        # make sure coords are set equal to call_args
        if not self._sub_fields:
            assert len(self.filter_coords(infile_call_args, result)) == 0, \
                f"{self.filter_coords(infile_call_args, result)} not present in result"
        else:
            for field in vars(result):
                assert len(self.filter_coords(infile_call_args, getattr(result, field))) == 0

    def filter_callargs(self, result, callargs):
        # filter to what function was called with
        if not self._sub_fields:
            result = self.filter_data(result, callargs)
        else:
            for field in vars(result):
                setattr(result, field, self.filter_data(getattr(result, field), callargs))
        return result

    def filter_coords(self, call_args, result):
        for key, value in call_args.items():
            assert is_iterable(value)
        combinations = [dict(zip(call_args, values)) for values in itertools.product(*call_args.values())]
        uncomputed_combinations = []
        for combination in combinations:
            combination_result = result
            combination_result = self.filter_data(combination_result, combination, single_value=True)
            if combination_result.size == 0:
                uncomputed_combinations.append(combination)
        if len(uncomputed_combinations) == 0:
            return {}
        return self._combine_call_args(uncomputed_combinations)

    def filter_data(self, data, coords, single_value=False):
        for coord, coord_value in coords.items():
            if not hasattr(data, coord):
                raise ValueError("Coord not found in data: {}".format(coord))
            # when called with a combination instantiation, coord_value will be a single item; iterable for check
            indexer = np.array([(val == coord_value) if single_value or not is_iterable(coord_value)
                                else (val in coord_value) for val in data[coord].values])
            coord_dims = data[coord].dims
            dim_indexes = {dim: slice(None) if dim not in coord_dims else np.where(indexer)[0]
                           for dim in data.dims}
            data = data.isel(**dim_indexes)
        data = data.sortby([self._build_sort_array(coord, coord_value, data)
                            for coord, coord_value in coords.items()
                            if is_iterable(coord_value) and len(coord_value) > 1])
        return data

    def _combine_call_args(self, uncomputed_combinations):
        call_args = defaultdict(list)
        for combination in uncomputed_combinations:
            for key, value in combination.items():
                call_args[key].append(value)
        return call_args

    def _build_sort_array(self, coord, coord_value, data):
        dims = data[coord].dims
        assert len(dims) == 1
        if isinstance(coord_value, tuple):
            coord_value = list(coord_value)
        s = xr.DataArray(list(range(len(coord_value))), [(coord, coord_value)])
        if dims[0] == coord:
            return s
        return s.stack(**{dims[0]: [coord]})


    # --- S3 backend -------------------------------------------------------
    # Routed to from is_stored/load/save when RESULTCACHING_S3_BUCKET is set.
    # Config is read per call rather than cached in __init__ because decoration
    # happens at import time, long before the environment is what it will be.

    def _s3_config(self):
        try:
            max_gb = float(os.getenv(_S3_MAX_GB_VAR, _S3_DEFAULT_MAX_GB))
        except ValueError:
            max_gb = _S3_DEFAULT_MAX_GB
        return {
            'bucket': os.getenv(_S3_BUCKET_VAR, '').strip(),
            'prefix': os.getenv(_S3_PREFIX_VAR, 'result_caching').strip('/'),
            'epoch': os.getenv(_S3_EPOCH_VAR, '1').strip(),
            'max_bytes': max_gb * (1024 ** 3),
        }

    def s3_key(self, function_identifier):
        config = self._s3_config()
        return f"{config['prefix']}/epoch={config['epoch']}/{function_identifier}.nc"

    def _s3_staging_path(self, function_identifier):
        """Scratch path for the netCDF file in transit.

        netCDF needs a real file, so bytes land on local disk on the way to or
        from S3. Kept under RESULTCACHING_HOME so container disk budgeting does
        not have to learn about a second location.
        """
        return os.path.join(self._storage_directory, '_s3_staging', function_identifier + '.nc')

    def _s3_read_manifest(self, client, config, key):
        try:
            response = client.get_object(Bucket=config['bucket'], Key=key + '.manifest.json')
            return json.loads(response['Body'].read())
        except Exception:
            # Absent manifest, no permission, malformed JSON -- all mean the
            # entry is not usable, which is a miss.
            return None

    def _s3_is_stored(self, function_identifier):
        client = _s3_client()
        if client is None:
            return False
        config = self._s3_config()
        key = self.s3_key(function_identifier)
        manifest = self._s3_read_manifest(client, config, key)
        if manifest is None:
            return False
        if manifest.get('schema_version') != _MANIFEST_SCHEMA_VERSION:
            self._logger.debug("s3://%s/%s has schema_version %s, expected %s",
                               config['bucket'], key, manifest.get('schema_version'),
                               _MANIFEST_SCHEMA_VERSION)
            return False
        return True

    def _s3_load(self, function_identifier):
        client = _s3_client()
        config = self._s3_config()
        local_path = self._s3_staging_path(function_identifier)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        key = self.s3_key(function_identifier)
        client.download_file(config['bucket'], key, local_path)
        manifest = self._s3_read_manifest(client, config, key) or {}
        # Always netCDF here, whatever RESULTCACHING_FORMAT says: a shared cache
        # has no reason to carry the version-fragile pickle format.
        return _netcdf_load(local_path, target_class=_resolve_class(manifest.get('class')))

    def _s3_save(self, result, function_identifier):
        client = _s3_client()
        if client is None:
            return
        config = self._s3_config()
        local_path = self._s3_staging_path(function_identifier)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            _netcdf_save(result, local_path)
            size = os.path.getsize(local_path)
            if size > config['max_bytes']:
                self._logger.info("Not caching %s: %.1f GB exceeds the %.1f GB cap (%s)",
                                  function_identifier, size / (1024 ** 3),
                                  config['max_bytes'] / (1024 ** 3), _S3_MAX_GB_VAR)
                return
            key = self.s3_key(function_identifier)
            # Data first, manifest second: a read requires the manifest, so an
            # interrupted write reads as a miss rather than a corrupt hit.
            client.upload_file(local_path, config['bucket'], key)
            client.put_object(Bucket=config['bucket'], Key=key + '.manifest.json',
                              Body=json.dumps(self._s3_manifest(result, size, config),
                                              sort_keys=True).encode())
        except Exception:
            # Caching is an optimisation; a write failure must not lose the
            # result that was just computed.
            self._logger.warning("Could not write %s to S3 cache", function_identifier,
                                 exc_info=True)
        finally:
            if os.path.isfile(local_path):
                os.remove(local_path)

    def _s3_manifest(self, result, size, config):
        import numpy
        return {
            'schema_version': _MANIFEST_SCHEMA_VERSION,
            'packages': {'xarray': xr.__version__, 'pandas': pd.__version__,
                         'numpy': numpy.__version__},
            'dtype': str(getattr(result, 'dtype', '')),
            'shape': list(getattr(result, 'shape', ())),
            'bytes': size,
            'epoch': config['epoch'],
            'class': _describe_class(result),
        }


    # --- format selection -------------------------------------------------
    # Overridden from _DiskStorage so a shared cache can use a format that
    # survives a pandas/xarray upgrade. See use_netcdf_format().

    def storage_path(self, function_identifier):
        extension = '.nc' if use_netcdf_format() else '.pkl'
        return os.path.join(self._storage_directory, function_identifier + extension)

    def save_file(self, result, savepath_part):
        if not use_netcdf_format():
            return super().save_file(result, savepath_part)
        # Flattens MultiIndex coords before writing -- the part of the
        # xarray/pandas boundary that does not round-trip across versions.
        _netcdf_save(result, savepath_part)

    def save(self, result, function_identifier):
        if s3_enabled():
            return self._s3_save(result, function_identifier)
        super().save(result, function_identifier)
        if use_netcdf_format():
            # After super().save() has renamed the temp file into place, so the
            # manifest can record the final size.
            try:
                _write_manifest(self.storage_path(function_identifier), result)
            except Exception:
                logging.getLogger(__name__).debug("Could not write cache manifest", exc_info=True)

    def load(self, function_identifier):
        if s3_enabled():
            return self._s3_load(function_identifier)
        return super().load(function_identifier)

    def load_file(self, path):
        if not use_netcdf_format():
            return super().load_file(path)
        return _netcdf_load(path, target_class=_class_from_manifest(_manifest_path(path)))

    def is_stored(self, function_identifier):
        if s3_enabled():
            return self._s3_is_stored(function_identifier)
        if not super().is_stored(function_identifier):
            return False
        if not use_netcdf_format():
            return True
        storage_path = self.storage_path(function_identifier)
        if not _manifest_is_usable(storage_path):
            return False
        # Final guard: an entry that cannot actually be opened must read as a
        # miss, never as an error. This is what makes an environment upgrade
        # cost one recomputation instead of failing the run.
        try:
            self.load_file(storage_path).close()
        except Exception:
            logging.getLogger(__name__).warning(
                f"Cache entry {storage_path} is unreadable in this environment; recomputing.")
            return False
        return True


# --------------------------------------------------------------------------
# S3-backed xarray storage
#
# Containers are ephemeral, so a local cache is worthless to them: the point of
# caching activations is that an entry outlives the container that wrote it.
# This backend keeps the netCDF + manifest format above and moves the bytes to
# S3.
#
# boto3 is an optional dependency (`pip install result_caching[s3]`); importing
# this module never requires it.
# --------------------------------------------------------------------------





class _MemoryStorage(_Storage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = dict()

    def save(self, result, function_identifier):
        self.cache[function_identifier] = result

    def is_stored(self, function_identifier):
        return function_identifier in self.cache

    def load(self, function_identifier):
        return self.cache[function_identifier]


def is_iterable(x):
    try:
        iter(x)
        if isinstance(x, str):
            return False
        return True
    except TypeError:
        return False


def _fullname(obj):
    return obj.__module__ + "." + obj.__class__.__name__


def get_calling_function():
    """
    finds the calling function in many decent cases.

    Note: this function is unreliable during debugging.
    """
    # https://stackoverflow.com/a/39079070/2225200
    fr = inspect.stack()[1][0]
    co = fr.f_code
    for get in (
            lambda: fr.f_globals[co.co_name],
            lambda: getattr(fr.f_locals['self'], co.co_name),
            lambda: getattr(fr.f_locals['cls'], co.co_name),
            lambda: fr.f_back.f_locals[co.co_name],  # nested
            lambda: fr.f_back.f_locals['func'],  # decorators
            lambda: fr.f_back.f_locals['meth'],
            lambda: fr.f_back.f_locals['f'],
    ):
        try:
            func = get()
        except (KeyError, AttributeError):
            pass
        else:
            if func.__code__ == co:
                return func
    raise AttributeError("func not found")


cache = _MemoryStorage
store = _DiskStorage
store_dict = _DictStorage
store_xarray = _XarrayStorage
store_xarray_s3 = _XarrayStorage
store_netcdf = _NetcdfStorage
