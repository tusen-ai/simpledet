import types
import inspect


class NoThrowBase:
    def __getattr__(self, item):
        return None


class NoThrowMeta(type):
    def __getattr__(self, item):
        return None


def patch_config_as_nothrow(instance):
    if "NoThrow" in [instance.__name__, instance.__class__.__name__]:
        return instance

    if type(instance) == type:
        instance = types.new_class(instance.__name__ + "NoThrow", (instance, ), dict(metaclass=NoThrowMeta))
        for (k, v) in inspect.getmembers(instance):
            if not k.startswith("__") and type(v) == type:
                type.__setattr__(instance, k, patch_config_as_nothrow(v))
    else:
        for (k, v) in inspect.getmembers(instance.__class__):
            if not k.startswith("__") and type(v) == type:
                type.__setattr__(instance.__class__, k, patch_config_as_nothrow(v))
        instance.__class__ = type(instance.__class__.__name__ + "NoThrow", (instance.__class__, NoThrowBase), {})

    return instance


if __name__ == "__main__":
    class A:
        a = 1

    A = patch_config_as_nothrow(A)
    assert A.non_exist is None
    assert A.a == 1

    class B:
        b = 1
        class B1:
            b1 = 2

    B = patch_config_as_nothrow(B)
    assert B.non_exist is None
    assert B.B1.non_exist is None
    assert B.b == 1
    assert B.B1.b1 == 2

    class B:
        b = 1
        class B1:
            b1 = 2
            def b1f():
                return 3

    b = B()
    b = patch_config_as_nothrow(b)
    assert b.non_exist is None
    assert b.B1.non_exist is None
    assert b.b == 1
    assert b.B1.b1 == 2
    assert b.B1.b1f() == 3
