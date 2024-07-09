import numpy as np

class distance(object):
    def __init__(self, need_complete_data = False):
        self.need_complete_data = need_complete_data

    def calculate(self, X, Y):
        raise NotImplementedError()

    def preset_x(self, X):
        self.X = X

    def reset_global_vars(self):
        raise NotImplementedError()