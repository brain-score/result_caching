[![Build Status](https://travis-ci.org/brain-score/result_caching.svg?branch=master)](https://travis-ci.org/brain-score/result_caching)

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
|                       | * `'module1,module2.submodule.function`' to disable loading and saving of function identifiers starting with one of the specifiers (e.g. `module1.function` will not be considered) |
