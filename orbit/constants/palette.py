from enum import Enum
import seaborn as sns
import matplotlib.colors as clr


class KTRPalette(Enum):
    KNOTS_SEGMENT = '#276ef1'
    KNOTS_REGION = '#05A357'


class OrbitPalette(Enum):
    # Qualitative Palette
    # single brand colors
    BLACK = '#000000'
    BLUE = '#276EF1'
    BLUE600 = '#174291'
    GREEN = '#05A357'
    GREEN600 = '#03582F'
    YELLOW = '#FFC043'
    YELLOW400 = '#FFC043'
    RED = '#E11900'
    BROWN = '#99644C'
    ORANGE = '#ED6E33'
    PURPLE = '#7356BF'
    WHITE = '#FFFFFF'

    # Sequential Palette
    BLACK_GRADIENT = clr.LinearSegmentedColormap.from_list('custom', [WHITE, BLACK], N=300)
    BLUE_GRADIENT = clr.LinearSegmentedColormap.from_list('custom', [WHITE, BLUE, BLACK], N=300)
    GREEN_GRADIENT = clr.LinearSegmentedColormap.from_list('custom', [WHITE, GREEN, BLACK], N=300)
    YELLOW_GRADIENT = clr.LinearSegmentedColormap.from_list('custom', [WHITE, YELLOW, BLACK], N=300)
    RED_GRADIENT = clr.LinearSegmentedColormap.from_list('custom', [WHITE, RED, BLACK], N=300)
    PURPLE_GRADIENT = clr.LinearSegmentedColormap.from_list('custom', [WHITE, PURPLE, BLACK], N=300)

    # Diverging Palette -  blue green yellow orange red
    RAINBOW = clr.LinearSegmentedColormap.from_list(
        'custom', ['#276EF1', '#05A357', '#FFC043',  '#ED6E33', '#E11900'],
        N=300)


class PredictionPaletteClassic(Enum):
    """
    str
    """
    # black
    ACTUAL_OBS = '#000000'
    # blue
    PREDICTION_LINE = '#276EF1'
    PREDICTION_INTERVAL = '#276EF1'
    # black
    HOLDOUT_VERTICAL_LINE = '#000000'
    # yellow
    TEST_OBS = '#FFC043'
