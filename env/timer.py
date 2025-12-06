import time
import functools
from contextlib import contextmanager

@contextmanager
def timer(description="Operation"):
    start = time.time()
    yield
    end = time.time()
    print(f"{description}: {end - start:.4f} seconds")

def time_method(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}: {end - start:.4f} seconds")
        return result
    return wrapper