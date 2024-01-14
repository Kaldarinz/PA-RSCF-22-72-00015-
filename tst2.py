from random import random
import time
import numpy as np
from collections import Counter
import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


# class PointSet(object):
#     def __init__(self, numpoints):
#         self.points = [Point(random(), random()) for _ in range(numpoints)]

#     def update(self):
#         for point in self.points:
#             point.x += random() - 0.5
#             point.y += random() - 0.5

# class Point(object):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

# start = time.time()
# points = PointSet(100000)
# point = points.points[10]

# for _ in range(1000):
#     points.update()
# stop = time.time()
# print(stop-start)

class PointSet(object):
    def __init__(self, numpoints):
        self.coords = np.random.random((numpoints, 2))
        self.points = [Point(i, self.coords) for i in range(numpoints)]

    def update(self):
        """Update along a random walk."""
        # The "+=" is crucial here... We have to update "coords" in-place, in
        # this case. 
        self.coords += np.random.random(self.coords.shape) - 0.5

class Point(object):
    def __init__(self, i, coords):
        self.i = i
        self.coords = coords

    @property
    def x(self):
        return self.coords[self.i,0]

    @property
    def y(self):
        return self.coords[self.i,1]

start = time.time()
points = PointSet(1000)
point = points.points[10]

for _ in range(1000):
    points.update()
stop = time.time()
print(stop-start)

p = points.points[10]
print(f'{p.i=}')
print(f'{p.coords[p.i,0]=}')
print(f'{p.x=}')
if p.x is p.coords[p.i,0]:
    print('ok')

a = [10]
b = a[0]
c = a[0]
if b is a[0]:
    print('b is c')