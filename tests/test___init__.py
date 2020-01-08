import os
import tempfile
from collections import defaultdict

import numpy as np
import pytest
import xarray as xr

from result_caching import store_xarray, store, cache, store_dict, get_function_identifier, NotCachedError


class TestFunctionIdentifier:
    def test_noargs(self):
        identifier = get_function_identifier(TestFunctionIdentifier.test_noargs, dict())
        assert identifier == 'test___init__.test_noargs'

    def test_two_ints(self):
        identifier = get_function_identifier(TestFunctionIdentifier.test_two_ints, dict(a=1, b=2))
        assert identifier == 'test___init__.test_two_ints/a=1,b=2'

    def test_slashes(self):
        identifier = get_function_identifier(TestFunctionIdentifier.test_slashes, dict(path='/local/user/msch/hello'))
        assert identifier == 'test___init__.test_slashes/path=_local_user_msch_hello'


class TestDictStore:
    def test_same(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir
            function_called = False

            @store_dict(identifier_ignore=['x'], dict_key='x')
            def func(x):
                nonlocal function_called
                assert not function_called
                function_called = True
                return {_x: _x for _x in x}

            def test():
                result = func([1])
                assert isinstance(result, dict)
                assert result[1] == 1

            test()
            # second call returns same thing and doesn't actually call function again
            test()

    def test_complimentary(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir

            @store_dict(identifier_ignore=['x'], dict_key='x')
            def func(x):
                return {_x: _x for _x in x}

            assert func([1]) == {1: 1}
            assert func([2]) == {2: 2}

    def test_missing_coord(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir

            @store_dict(identifier_ignore=['x'], dict_key='x')
            def func(x):
                return {}

            with pytest.raises(ValueError):
                func([1])

    def test_combined(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir

            expected_x = None

            @store_dict(identifier_ignore=['x'], dict_key='x')
            def func(x):
                assert len(x) == 1 and x[0] == expected_x
                return {_x: _x for _x in x}

            expected_x = 1
            assert func([1]) == {1: 1}
            expected_x = 2
            combined = func([1, 2])
            assert combined == {1: 1, 2: 2}


class TestXarrayStore:
    def test_same(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir
            function_called = False

            @store_xarray(identifier_ignore=['x'], combine_fields=['x'])
            def func(x, base=1):
                nonlocal function_called
                assert not function_called
                function_called = True
                return xr.DataArray(x, coords={'x': x, 'base': ('x', [base])}, dims='x')

            def test():
                result = func([1])
                assert isinstance(result, xr.DataArray)
                assert result == 1

            test()
            # second call returns same thing and doesn't actually call function again
            test()

    def test_complimentary(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir

            @store_xarray(identifier_ignore=['x'], combine_fields=['x'])
            def func(x, base=1):
                return xr.DataArray(x, coords={'x': x, 'base': ('x', [base])}, dims='x')

            np.testing.assert_array_equal(func([1]), 1)
            np.testing.assert_array_equal(func([2]), 2)

    def test_missing_coord(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir

            @store_xarray(identifier_ignore=['x'], combine_fields=['x'])
            def func(x, base=1):
                return xr.DataArray(0)

            with pytest.raises(ValueError):
                func([1])

    def test_combined(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir

            expected_x = None

            @store_xarray(identifier_ignore=['x'], combine_fields=['x'])
            def func(x, base=1):
                assert len(x) == 1 and x[0] == expected_x
                return xr.DataArray(x, coords={'x': x, 'base': ('x', [base])}, dims='x')

            expected_x = 1
            np.testing.assert_array_equal(func([1]), 1)
            expected_x = 2
            combined = func([1, 2])
            np.testing.assert_array_equal(combined, [1, 2])


class TestStore:
    # storage_configuration (i.e. `os.environ['RESULTCACHING_HOME'] = ...`) is tested implicitly in other tests.

    def test_basic(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir

            function_called = False

            @store()
            def func(x):
                nonlocal function_called
                assert not function_called
                function_called = True
                return x

            assert func(1) == 1
            assert os.path.isfile(os.path.join(storage_dir, 'test___init__.func', 'x=1.pkl'))
            # second call returns same thing and doesn't actually call function again
            assert func(1) == 1

    def test_object(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir

            function_called = False

            class C(object):
                @store()
                def f(self, x):
                    nonlocal function_called
                    assert not function_called
                    function_called = True
                    return x

            c = C()
            assert c.f(1) == 1
            assert os.path.isfile(os.path.join(storage_dir, 'test___init__.C.f', 'x=1.pkl'))
            # second call returns same thing and doesn't actually call function again
            assert c.f(1) == 1

    def test_object_custom_repr(self):
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir

            function_called = False

            class C(object):
                @store()
                def f(self, x):
                    nonlocal function_called
                    assert not function_called
                    function_called = True
                    return x

                def __repr__(self):
                    return "custom_repr"

            c = C()
            assert c.f(1) == 1
            assert os.path.isfile(os.path.join(storage_dir, 'test___init__.C(custom_repr).f', 'x=1.pkl'))
            # second call returns same thing and doesn't actually call function again
            assert c.f(1) == 1

    def test_disable_all(self):
        previous_disable_value = os.getenv('RESULTCACHING_DISABLE', '')
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir
            os.environ['RESULTCACHING_DISABLE'] = '1'

            function_calls = 0

            @store()
            def func(x):
                nonlocal function_calls
                function_calls += 1
                return x

            assert func(1) == 1
            assert function_calls == 1
            assert not os.listdir(storage_dir)
            assert func(1) == 1
            assert function_calls == 2

        os.environ['RESULTCACHING_DISABLE'] = previous_disable_value

    def test_disable_specific(self):
        previous_disable_value = os.getenv('RESULTCACHING_DISABLE', '')
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir
            os.environ['RESULTCACHING_DISABLE'] = 'test___init__.func1'

            function_calls = defaultdict(lambda: 0)

            @store()
            def func1(x):
                nonlocal function_calls
                function_calls[1] += 1
                return x

            @store()
            def func2(x):
                nonlocal function_calls
                function_calls[2] += 1
                return x

            assert func1(1) == 1
            assert function_calls[1] == 1
            assert not os.listdir(storage_dir)
            assert func1(1) == 1
            assert function_calls[1] == 2

            assert func2(1) == 1
            assert function_calls[2] == 1
            assert func2(1) == 1
            assert function_calls[2] == 1

        os.environ['RESULTCACHING_DISABLE'] = previous_disable_value

    def test_disable_module(self):
        previous_disable_value = os.getenv('RESULTCACHING_DISABLE', '')
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir
            os.environ['RESULTCACHING_DISABLE'] = 'test___init__'

            function_calls = defaultdict(lambda: 0)

            @store()
            def func1(x):
                nonlocal function_calls
                function_calls[1] += 1
                return x

            @store()
            def func2(x):
                nonlocal function_calls
                function_calls[2] += 1
                return x

            assert func1(1) == 1
            assert func1(1) == 1
            assert function_calls[1] == 2
            assert func2(1) == 1
            assert func2(1) == 1
            assert function_calls[2] == 2
            assert not os.listdir(storage_dir)

        os.environ['RESULTCACHING_DISABLE'] = previous_disable_value

    def test_cachedonly_specific(self):
        previous_cached_value = os.getenv('RESULTCACHING_CACHEDONLY', '')
        with tempfile.TemporaryDirectory() as storage_dir:
            os.environ['RESULTCACHING_HOME'] = storage_dir

            @store()
            def func1(x):
                return x

            @store()
            def func2(x):
                return x

            # when allowing only cached results from func2, func1 should work, but func2 should err
            os.environ['RESULTCACHING_CACHEDONLY'] = 'test___init__.func2'
            assert func1(1) == 1
            with pytest.raises(NotCachedError):
                func2(2)

            # when allow reruns, func2 should work again
            os.environ['RESULTCACHING_CACHEDONLY'] = ''
            assert func2(2) == 2

            # when now only allowing cached results again, func2 should work because results are already cached
            os.environ['RESULTCACHING_CACHEDONLY'] = 'test___init__.func2'
            assert func2(2) == 2

        os.environ['RESULTCACHING_CACHEDONLY'] = previous_cached_value


class TestCache:
    def test(self):
        function_called = False

        @cache()
        def func(x):
            nonlocal function_called
            assert not function_called
            function_called = True
            return x

        assert func(1) == 1
        # second call returns same thing and doesn't actually call function again
        assert func(1) == 1
