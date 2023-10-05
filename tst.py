from dataclasses import dataclass, replace

@dataclass
class Tst:
    a: int = 0
    b: float = 1.0

dclas = Tst()
print(dclas)
dclas = replace(dclas, **{'a': 1, 'b': 3.0})
print(dclas)