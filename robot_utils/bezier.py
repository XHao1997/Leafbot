import numpy as np
from scipy.special import comb
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
def bezier_curve(points, nTimes=1000):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.

    points should be a list of lists, or list of tuples
    such as [ [1,1], 
                [2,3], 
                [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def interpolate_orientation(start_orientation, end_orientation, num_points):
    # start_rot = Rotation.from_matrix(start_orientation).as_euler('xyz')
    # end_rot = Rotation.from_matrix(end_orientation).as_euler('xyz')
    # start_rot = Rotation.as_euler('xyz')
    # end_rot = Rotation.as_euler('xyz')
    key_rots = Rotation.from_euler('xyz', np.stack((start_orientation, end_orientation)))
    
    slerp = Slerp([0, 1], key_rots)
    interpolated_rotations = slerp(np.linspace(0, 1, num_points))
    orientations = interpolated_rotations.as_euler('xyz')

    return orientations