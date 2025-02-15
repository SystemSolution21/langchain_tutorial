from typing import Callable

addition: Callable[[int, int], int] = lambda x, y: x + y
print(addition(x=3, y=2))
