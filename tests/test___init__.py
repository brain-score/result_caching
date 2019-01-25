import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from result_caching import store_xarray, store, cache, store_dict


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
