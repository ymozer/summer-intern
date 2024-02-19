from functools import wraps
import time
import asyncio


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap


def async_timeit(func, *args, **params):
    async def process(func, *args, **params):
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **params)
        else:
            print('this is not a coroutine')
            return func(*args, **params)

    async def helper(*args, **params):
        start = time.time()
        result = await process(func, *args, **params)
        end = time.time() - start
        sec = "{:.2f}".format(end)
        # if param are passed to the function and param is 'id'
        if len(args) > 1 and 'id' in params:
            print(f'\n>>> Time took for {func.__name__} {sec} with id: {args[1]}')
        print(f'>>> Time took for {func.__name__} function {sec} sec')
        return result
    return helper
