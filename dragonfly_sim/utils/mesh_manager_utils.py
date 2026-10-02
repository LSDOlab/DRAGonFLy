import numpy as np


def airfoil_inner_bdry_function(x):
    """
    Input: x is an array of size (n_dimensions, n_points)
    """
    y_limits = (-0.08, 0.08)
    norm_test = np.linalg.norm(x, axis=0) <= 1.1
    y_limit_test = ((y_limits[0] <= x[1, :]) & (x[1, :] <= y_limits[1]))
    return (norm_test & y_limit_test)


def wing_inner_bdry_function(x):
    """
    Input: x is an array of size (n_dimensions, n_points)
    """
    x_limits = (-1e-10, 9.1 + 1e-10)
    y_limits = (1e-10, 14.1 + 1e-10)
    z_limits = (-0.4, 0.4)

    x_limit_test = ((x_limits[0] <= x[0, :]) & (x[0, :] <= x_limits[1]))
    y_limit_test = ((y_limits[0] <= x[1, :]) & (x[1, :] <= y_limits[1]))
    z_limit_test = ((z_limits[0] <= x[2, :]) & (x[2, :] <= z_limits[1]))
    return (x_limit_test & y_limit_test & z_limit_test)
