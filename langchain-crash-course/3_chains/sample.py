from typing import Callable

addition: Callable[[int, int], int] = lambda x, y: x + y
print(f"{addition(x=3, y=2) = }")
