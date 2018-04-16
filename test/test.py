
def _func(a, b, c, d, e):
    print("a:", a)
    print("b:", b)
    print("c:", c)
    print("d:", d)
    print("e:", e)

def func(a, b):
    return lambda *args, **kwargs: _func(a,
                                         b,
                                         *args,
                                         **kwargs)

f = func(1, 2)
f(3, 4, 5)