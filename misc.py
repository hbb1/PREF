import numpy as np
import torch
import functools

to_float = lambda x: float(x)
# to_float = functools.partial(function=map(lambda x: float(x), ))
blender_aabb ={
    'chair':     list(map(to_float, ['-0.77', '-0.74', '-1.10', '0.70', '0.74', '1.12'])),
    'drums':     list(map(to_float, ['-1.17', '-0.89', '-0.58', '1.05', '1.03', '0.98'])),
    'ficus':     list(map(to_float, ['-0.41', '-0.91', '-1.07', '0.60', '0.60', '1.17'])),
    'hotdog':    list(map(to_float, ['-1.24', '-1.33', '-0.25', '1.24', '1.17', '0.37'])),
    'lego':      list(map(to_float, ['-0.70', '-1.19', '-0.51', '0.72', '1.22', '1.24'])),
    'materials': list(map(to_float, ['-1.17', '-0.86', '-0.30', '1.12', '1.10', '0.25'])),
    'mic':       list(map(to_float, ['-1.31', '-0.96', '-0.81', '0.84', '1.12', '1.19'])),
    'ship':      list(map(to_float, ['-1.45', '-1.36', '-0.65', '1.43', '1.41', '0.77'])),
}

