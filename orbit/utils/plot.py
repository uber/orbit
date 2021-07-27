import pkg_resources

STYLE_FILE_NAME = 'plot_style.mplstyle'


def get_orbit_style():
    path = pkg_resources.resource_filename(
        'orbit',
        STYLE_FILE_NAME
    )
    return path