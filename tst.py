
attrs = {'speed': 5,
        'mass': 10}

class someCls():
    def __init__(self) -> None:
        pass
    def set_attrs(self, attrs:dict):
        for key, value in attrs.items():
            setattr(self,key,value)

osc = someCls()
osc.set_attrs(attrs)
print(osc.speed)