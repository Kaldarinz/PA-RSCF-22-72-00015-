class Tst:
    def __init__(self) -> None:
        self.b = 5

    def add_attr(self, name, val):
        setattr(self, name, val)

tst = Tst()
tst.add_attr('a', 10)
print(tst.__dict__)