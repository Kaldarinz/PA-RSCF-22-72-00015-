from dataclasses import dataclass, field

@dataclass
class Test:
    a: dict = field(default_factory=dict)

@dataclass
class Test2:
    a: dict = field(default_factory=dict)

cls1 = Test()
cls2 = Test2()

cls1.a.update({'one': 2})
cls2.a.update({'one': 1})

print(cls1)
print(cls2)