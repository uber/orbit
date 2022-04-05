import pkg_resources
import functools
import logging
from matplotlib import pyplot as plt

STYLE_FILE_NAME = "plot_style"


def get_orbit_style():
    path = pkg_resources.resource_filename(
        "orbit", "stylelib/{}.mplstyle".format(STYLE_FILE_NAME)
    )
    return path


def orbit_style_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # check if arg is specified; if not, set True as default
        if "use_orbit_style" in kwargs.keys():
            use_orbit_style = kwargs["use_orbit_style"]
            del kwargs["use_orbit_style"]
        else:
            use_orbit_style = True

        # use orbit style plot if it is set to be True
        if use_orbit_style:
            orbit_style_path = get_orbit_style()
            try:
                with plt.style.context(orbit_style_path):
                    return func(*args, **kwargs)
            except:
                logging.info(
                    "Cannot find path:{}. Use default plot style.".format(
                        orbit_style_path
                    )
                )
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper
