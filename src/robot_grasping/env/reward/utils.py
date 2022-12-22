import numpy as np

def _scaled_clip(x:float, minv:float=0.0, maxv:float=1.0):
    return np.clip((x-minv)/(maxv-minv), 0.0, 1.0)

def value_increase_derivative_increase_p(x:float, minv:float=0.0, maxv:float=1.0)->float:
    cliped_x = _scaled_clip(x=x, minv=minv, maxv=maxv)
    return 1. - np.float_power(1. - cliped_x, 0.4)


def value_decrease_derivative_increase_p(x:float, minv:float=0.0, maxv:float=1.0)->float:
    cliped_x = _scaled_clip(x=x, minv=minv, maxv=maxv)
    return 1. - np.float_power(cliped_x, 0.4)


def value_increase_derivative_decrease_p(x:float, minv:float=0.0, maxv:float=1.0)->float:
    cliped_x = _scaled_clip(x=x, minv=minv, maxv=maxv)
    return np.float_power(cliped_x, 0.4)


def value_decrease_derivative_decrease_p(x:float, minv:float=0.0, maxv:float=1.0)->float:
    cliped_x = _scaled_clip(x=x, minv=minv, maxv=maxv)
    return np.float_power(1. - cliped_x, 0.4)




def value_increase_derivative_increase_n(x:float, minv:float=0.0, maxv:float=1.0)->float:
    return -1*value_decrease_derivative_decrease_p(x=x, minv=minv, maxv=maxv)


def value_decrease_derivative_increase_n(x:float, minv:float=0.0, maxv:float=1.0)->float:
    return -1*value_increase_derivative_decrease_p(x=x, minv=minv, maxv=maxv)


def value_increase_derivative_decrease_n(x:float, minv:float=0.0, maxv:float=1.0)->float:
    return -1*value_decrease_derivative_increase_p(x=x, minv=minv, maxv=maxv)


def value_decrease_derivative_decrease_n(x:float, minv:float=0.0, maxv:float=1.0)->float:
    return -1*value_increase_derivative_increase_p(x=x, minv=minv, maxv=maxv)
