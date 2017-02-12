import numpy as np
from numba import jitclass, int64, float32

import ruamel.yaml as yaml

_float = float32

_spec = [
    ('frame_support', int64),
    ('angle_support', _float),
    ('planar_support', _float),
    ('eps', _float),
    ('min_radius', _float),
    ('angle_res', _float),
    ('depth_res', _float),
    ('angle_thresh', _float),
    ('depth_thresh', _float),
    ('min_score', int64),
    ('inlier_thresh', _float),
    ('min_radius', _float),
    ('min_neighbors', _float),
    ('min_inlier', _float),
    ('max_inlier', _float),
    ('scale', _float),
]


@jitclass(_spec)
class Config(object):
    def __init__(self):
        self.frame_support = 600
        self.angle_support = 0.2
        self.planar_support = 0.25
        self.eps = 1e-3
        self.min_radius = 3*self.eps
        self.angle_res = np.deg2rad(2)
        self.depth_res = 5e-3

        self.depth_thresh = 2*self.depth_res
        self.angle_thresh = np.deg2rad(10)

        self.min_score = 20

        self.inlier_thresh = 0.05

        self.min_neighbors = 0.03
        self.min_radius = 80
        self.min_inlier = 0.2
        self.max_inlier = 1

        self.scale = 1


def create(**kwargs):
    config = Config()

    for key, value in kwargs.items():
        setattr(config, key, value)

    return config


def load(path, **kwargs):
    with open(path) as yfile:
        Y = yaml.safe_load(yfile)

        config = create(**Y)

        for key, value in kwargs.items():
            setattr(config, key, value)

        return config
