def cross_2D(x, y):
    return x[0] * y[1] - x[1] * y[0]


def moment_integrand(force_vec, lever_vec, dimensions):
    """
    The pitching-moment integrand: the component of (lever x force) about the
    axis the aerofoil/wing pitches around.

    This exists because cross_2D above is a genuinely 2D expression --
    x[0]*y[1] - x[1]*y[0] reads only components 0 and 1 -- so applying it
    unchanged to 3D vectors silently returns the moment about the VERTICAL
    axis (a yawing moment) rather than the pitching moment. The two coordinate
    conventions in use here disagree about which index is "vertical":

      2D: x = chordwise, y = vertical
          -> the pitch axis is z, and cross_2D(force, lever) is that
             component (up to the sign convention baked into compute_cm).
      3D: x = chordwise, y = spanwise, z = vertical
          (see the mesh note at the top of wing_opt.py)
          -> the pitch axis is y, i.e. (lever x force)_y = r_z*F_x - r_x*F_z.

    The 2D branch returns exactly the previous expression, so 2D results are
    unchanged bit-for-bit.
    """
    if dimensions == 2:
        return cross_2D(force_vec, lever_vec)
    elif dimensions == 3:
        return lever_vec[2]*force_vec[0] - lever_vec[0]*force_vec[2]
    else:
        raise ValueError(
            "moment_integrand supports 2 or 3 dimensions, got {}".format(dimensions))
