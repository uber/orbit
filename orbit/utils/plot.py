import pkg_resources
import functools
from matplotlib import pyplot as plt

STYLE_FILE_NAME = 'plot_style.mplstyle'


def get_orbit_style():
    path = pkg_resources.resource_filename(
        'orbit',
        STYLE_FILE_NAME
    )
    return path


def orbit_style_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "use_orbit_style" in kwargs.keys():
            use_orbit_style = kwargs['use_orbit_style']
            del kwargs['use_orbit_style']
        else:
            # default to be True if arg is not specified
            use_orbit_style = True
        if use_orbit_style:
            try:
                orbit_style = get_orbit_style()
                with plt.style.context(orbit_style):
                    return func(*args, **kwargs)
            except:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper
