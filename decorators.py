import functools
import time


def notify(msg_start="=== Test case execution start ===", msg_end="=== Test case execution end ===== \n"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(msg_start)
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(result)
            print(f"\nExecution Time: {(end_time - start_time)*1000:.1f} ms")
            print(msg_end)
        return wrapper
    return decorator

def notify_and_return(msg_start="=== Test case execution start ===", msg_end="=== Test case execution end ===== \n"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(msg_start)
            result = func(*args, **kwargs)
            print(result)
            print(msg_end)
            return result
        return wrapper
    return decorator