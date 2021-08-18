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


orbit_style = get_orbit_style()


def orbit_style_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            if not kwargs['use_orbit_style']:
                return func(*args, **kwargs)
            elif kwargs['use_orbit_style']:
                with plt.style.context(orbit_style):
                    return func(*args, **kwargs)
        except:
            kwargs['use_orbit_style'] = True
            with plt.style.context(orbit_style):
                return func(*args, **kwargs)

    return wrapper
