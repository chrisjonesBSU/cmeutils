from math import factorial

import matplotlib.pyplot as plt
import numpy as np


def get_histogram(data, normalize=False, bins="auto", x_range=None):
    """Bins a 1-D array of data into a histogram using
    the numpy.histogram method.

    Parameters
    ----------
    data : 1-D numpy.array, required
        Array of data used to generate the histogram
    normalize : boolean, default=False
        If set to true, normalizes the histogram bin heights
        by the sum of data so that the distribution adds
        up to 1
    bins : float, int, or str, default="auto"
        Method used by numpy to determine bin borders.
        Check the numpy.histogram docs for more details.
    x_range : (float, float), default = None
        The lower and upper range of the histogram bins.
        If set to None, then the min and max values of data are used.

    Returns
    -------
    bin_cetners : 1-D numpy.array
        Array of the bin center values
    bin_heights : 1-D numpy.array
        Array of the bin height values

    """
    bin_heights, bin_borders = np.histogram(data, bins=bins, range=x_range)
    if normalize is True:
        bin_heights = bin_heights/sum(bin_heights)
    bin_widths = np.diff(bin_borders)
    bin_centers = bin_borders[:-1] + bin_widths / 2
    return bin_centers, bin_heights


def threedplot(
    x,
    y,
    z,
    xlabel = "xlabel",
    ylabel = "ylabel",
    zlabel = "zlabel",
    plot_name = "plot_name"
    ):

    '''Plot a 3d heat map from 3 lists of numbers. This function is useful
    for plotting a dependent variable as a function of two independent variables.
    In the example below we use f(x,y)= -x^2 - y^2 +6 because it looks cool.

    Example
    -------

    We create two indepent variables and a dependent variable in the z axis and
    plot the result. Here z is the equation of an elliptic paraboloid.

    import random

    x = []
    for i in range(0,1000):
        n = random.uniform(-20,20)
        x.append(n)

    y = []
    for i in range(0,1000):
        n = random.uniform(-20,20)
        y.append(n)

    z = []
    for i in range(0,len(x)):
        z.append(-x[i]**2 - y[i]**2 +6)

    fig = threedplot(x,y,z)
    fig.show()

    Parameters
    ----------

    x,y,z : list of int/floats

    xlabel, ylabel, zlabel : str

    plot_name : str


    '''
    fig = plt.figure(figsize = (10, 10), facecolor = 'white')
    ax = plt.axes(projection='3d')
    ax.set_xlabel(xlabel,fontdict=dict(weight='bold'),fontsize=12)
    ax.set_ylabel(ylabel,fontdict=dict(weight='bold'),fontsize=12)
    ax.set_zlabel(zlabel,fontdict=dict(weight='bold'),fontsize=12)
    p = ax.scatter(x, y, z, c=z, cmap='rainbow', linewidth=7);
    plt.colorbar(p, pad = .1, aspect = 2.3)

    return fig


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smoothing filter used on distributions and potentials

    Parameters
    ----------
    y: 1D array-like, required
        The data sequence to be smoothed
    window_size : int, required
        The size of the smoothing window to use; must be an odd number
    order: int, required
        The polynomial order used by the smoothing filter
    deriv:
    rate:

    Returns
    -------
    1D array-like
        Smoothed array of y after passing through the filter

    """
    if not (isinstance(window_size, int) and isinstance(order, int)):
        raise ValueError("window_size and order must be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    b = np.mat(
        [
            [k ** i for i in order_range]
            for k in range(-half_window, half_window + 1)
        ]
    )
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")
