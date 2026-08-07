[![Build Status](http://brain-score-jenkins.com:8080/buildStatus/icon?job=result_caching%2Fresult_caching_daily_test)](http://brain-score-jenkins.com:8080/buildStatus/icon?job=result_caching%2Fresult_caching_daily_test)
# Result Caching
Stores function results so that they are not computed again on repetitive calls of the function with the same arguments.
Results can be stored either on disk or in memory.


## Quick setup
```
pip install git+https://github.com/mschrimpf/result_caching
```

## Usage example
```
from result_caching import store

@store()
def f(a, b):
	return a * b
	
y = f(1, 2)  # computed first time, stored on disk
y = f(1, 2)  # not computed again, loaded from disk
y = f(1, 3)  # computed again, different parameters
```

By default, results will be stored in `~/.result_caching`, this can be
changed through the environment variable `RESULTCACHING_HOME`.

`cache` will only hold results in memory and not write them to disk.

## Environment variables
| Variable | description |
|-----------------------|----------------------------------------------------------------------------------|
| RESULTCACHING_HOME | directory to cache results (benchmark ceilings) in, `~/.result_caching` by default |
| RESULTCACHING_DISABLE | * `'1'` to disable loading and saving of results, functions will be called directly |
|                       | * `'candidate_models.score_model,model_tools.activations`' to disable loading and saving of function identifiers starting with one of the specifiers separated by a comma (e.g. any package or function inside `model_tools.activations` will not be considered) |
| RESULTCACHING_CACHEDONLY | If enabled, raises an error when trying to run a function that does not have its result already cached (follows the same matching rules as `RESULTCACHING_DISABLE`) |
| RESULTCACHING_FORMAT | storage format for xarray results: `pickle` (default) or `netcdf`/`nc`. netCDF writes a `.nc` file plus a sidecar `.manifest.json` recording the class, dtype, shape, schema version and package versions |
| RESULTCACHING_S3_BUCKET | S3 bucket for xarray results. **Setting this is what enables the S3 backend** — unset means disk only. Requires the `s3` extra (`pip install result_caching[s3]`) |
| RESULTCACHING_S3_PREFIX | key prefix within the bucket, `result_caching` by default |
| RESULTCACHING_S3_EPOCH | prefix segment for bulk invalidation, `1` by default. Bumping it orphans every existing entry at once, which is cheaper than deleting them when a lifecycle rule will expire them anyway |
| RESULTCACHING_S3_MAX_GB | refuse to write entries larger than this, `50` by default. The size distribution has a long tail — one vision model projects to 146 GB against a p90 of 31 GB — and that tail is most of the storage bill for a fraction of the benefit |

### Storage formats

`pickle` (the default) is fast but tied to the exact pandas/xarray/numpy
versions that wrote it. `netcdf` is portable across environment upgrades, at the
cost of flattening MultiIndex coordinates on write and rebuilding them on load.

The default is deliberately still `pickle` so that existing warm caches stay
valid; opt in per environment rather than globally.

Either way an unreadable entry is treated as a **miss, never an error** — a
cache written before a dependency bump is recomputed rather than raising.

### S3 backend

Set `RESULTCACHING_S3_BUCKET` to make xarray results readable and writable
across machines, which is what makes the cache useful for ephemeral containers
where local disk does not survive the job.

Entries are always netCDF on S3 regardless of `RESULTCACHING_FORMAT`; the data
object is written **before** its manifest and a read requires the manifest, so
an interrupted write is indistinguishable from a miss and needs no temp-key
dance (S3 PUTs are already atomic).

boto3 is an optional dependency. If it is missing, the backend degrades to "no
cache" rather than breaking the import.
