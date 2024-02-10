from pprint import pprint
class Decr:
    def __get__(self, obj, type):
        print('Desr access.')

class DataDesr:
    def __set__(self, obj, val):
        print('Data descr access.')
    def __get__(self, obj, type):
        print('Val not set')

class A:
    a = Decr()

class B(A):
    b = 20

    def __init__(self) -> None:
        super().__init__()
        self.a = 30

inst = B()
print(inst.a)
