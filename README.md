# Caching
Stores function results so that they are not computed again on repetitive calls of the function with the same arguments.
Results can be stored either on disk or in memory.


## Usage example
```
from caching import store

@store()
def f(a, b):
	return a * b
	
y = f(1, 2)  # computed first time, stored on disk
y = f(1, 2)  # not computed again, loaded from disk
y = f(1, 3)  # computed again, different parameters
```
