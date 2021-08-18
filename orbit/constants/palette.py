from enum import Enum
import seaborn as sns
import matplotlib.colors as clr

class KTRPalette(Enum):
    KNOTS_SEGMENT = '#276ef1'
    KNOTS_REGION = '#5b91f5ff'


class OrbitPalette(Enum):
    # Qualitative Palette
    # single brand colors
    BLACK = '#000000'
    BLUE = '#276EF1'
    GREEN = '#05A357'
    YELLOW = '#FFC043'
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

    # Diverging Palette
    RAINBOW = clr.LinearSegmentedColormap.from_list(
        'custom', ['#FFFFFF', '#FFC043', '#3AA76D', '#276EF1', '#7356BF'],
        N=300)


class PredictionPaletteClassic(Enum):
    # black
    # actual_obs = '#000000'
    ACTUAL_OBS = '#000000'
    # dark_teal
    # prediction_line = '#12939A'
    # teal
    # prediction_interval = '#42999E'
    # blue
    PREDICTION_LINE = '#276EF1'
    PREDICTION_INTERVAL = '#276EF1'
    # blue 50%
    # prediction_range =
    # black dotted
      # # teal
    # preidction_range = '#42999E'
    # blue
    HOLDOUT_VERTICAL_LINE = '#000000'
    # holdout_vertical_line = '#1f77b4'
    # orange
    # test_obs = '#FF8C00'
    TEST_OBS = '#FFC043'


# class QualitativePalette(Enum):
#     """
#     Palette for visualizing discrete categorical data
#     """
#     Rainbow8 = ["#ffadadff", "#ffd6a5ff", "#fdffb6ff", "#caffbfff", "#9bf6ffff", "#a0c4ffff",
#                 "#bdb2ffff", "#ffc6ffff"]
#     # for time-series plot
#     Line4 = ["#e6c72b", "#2be669", "#2b4ae6", "#e62ba8"]
#     PostQ = ['#1fc600', '#ff4500']
#
#     # large amount of stacking series
#     Stack = ["#12939A", "#F15C17", "#DDB27C", "#88572C", "#FF991F", "#DA70BF", "#125C77",
#              "#4DC19C", "#776E57", "#17B8BE", "#F6D18A", "#B7885E", "#FFCB99", "#F89570",
#              "#829AE3", "#E79FD5", "#1E96BE", "#89DAC1", "#B3AD9E"]
#     # bar plot
#     Bar5 = ["#ef476fff", "#ffd166ff", "#06d6a0ff", "#118ab2ff", "#073b4cff"]
#
#     # single colors
#     teal = '#008080'
#     dark_teal = '#003f5c'
#     blue = '#0000FF'
#     mid_blue = '#4c72b0'
#     dark_blue = '#2f4b7c'
#     purple = 'a05195'
#     dark_purple = '#665191'
#     coral = '#f95d6a'
#     red = '#FF0000'
#     yellow = '#FFFF00'
#     bee_yellow = '#ffa600'
#     orange = '#dd8452'
#     gray = '#cccccc'
#     black = '#000000'
#     green = '#2eb82e'
#     dark_green = '#145214'
#
#     # paired color list
#     paired_colors = sns.color_palette("Paired")





# class SequentialPalette(Enum):
#     # Gradient colors of mono hue for numeric and inherently ordered values
#     # good for correlation plots or maps etc
#     # also colorblind friendly
#     Blue10 = ['#edf5ff', '#d0e2ff', '#a6c8ff', '#78a9ff', '#4589ff', '#0f62fe', '#0043ce', '#002d9c', '#001d6c',
#               '#001141']
#     Cyan10 = ['#e5f6ff', '#bae6ff', '#82cfff', '#33b1ff', '#1192e8', '#0072c3', '#00539a', '#003a6d', '#012749',
#               '#1c0f30']
#     Teal10 = ['#d9fbfb', '#9ef0f0', '#3ddbd9', '#08bdba', '#009d9a', '#007d79', '#005d5d', '#004144', '#022b30',
#               '#081a1c']
#     Orange10 = ['#fff2e6', '#ffd9b3', '#ffbf80', '#ffa64d', '#ff8c1a', '#e67300', '#b35900', '#804000', '#4d2600',
#                 '#1a0d00']
#
#     # continuous sequential color palette
#     # blue teal theme
#     Seafoam = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
#     # orange yellow theme
#     Sunshine = sns.color_palette("YlOrBr", as_cmap=True)
#     # green theme
#     Forest = sns.cubehelix_palette(start=1.5, rot=.1, as_cmap=True)
#     # coral theme
#     Rose = sns.cubehelix_palette(start=1.2, rot=-.2, as_cmap=True)


# class DivergingPalette(Enum):
#     # diverging colors with two hues for inherently ordered values
#     # good for numeric variable has a meaningful central value, like zero. The colors gradients from the center
#     # to the right and the left.
#     # green and red theme
#     Watermelon = sns.diverging_palette(120, 20, as_cmap=True)
#     # blue and red theme
#     Unclesam = sns.diverging_palette(260, 20, as_cmap=True)
#     # # dark teal purple orange theme
#     Sunrise = sns.color_palette("magma", as_cmap=True)
#     # # continuous palattes
#     Rainbow = sns.color_palette('Spectral', as_cmap=True)




