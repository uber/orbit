from enum import Enum


class QualitativePalette(Enum):
    """
    Palette for visualizing discrete categorical data
    """
    Rainbow8 = ["#ffadadff", "#ffd6a5ff", "#fdffb6ff", "#caffbfff", "#9bf6ffff", "#a0c4ffff",
                "#bdb2ffff", "#ffc6ffff"]
    # for time-series plot
    Line4 = ["#e6c72b", "#2be669", "#2b4ae6", "#e62ba8"]
    PostQ = ['#1fc600', '#ff4500']
    # large amount of stacking series
    Stack = ["#12939A", "#F15C17", "#DDB27C", "#88572C", "#FF991F", "#DA70BF", "#125C77",
             "#4DC19C", "#776E57", "#17B8BE", "#F6D18A", "#B7885E", "#FFCB99", "#F89570",
             "#829AE3", "#E79FD5", "#1E96BE", "#89DAC1", "#B3AD9E"]
    # bar plot
    Bar5 = ["#ef476fff", "#ffd166ff", "#06d6a0ff", "#118ab2ff", "#073b4cff"]


class KTRPalette(Enum):
    KNOTS_SEGMENT = '#276ef1'
    KNOTS_REGION = '#5b91f5ff'


class SequentialPalette(Enum):
    # Gradient colors of mono hue for numeric and inherently ordered values
    # good for correlation plots or maps etc
    # also colorblind friendly
    Blue10 = ['#edf5ff', '#d0e2ff', '#a6c8ff', '#78a9ff', '#4589ff', '#0f62fe', '#0043ce', '#002d9c', '#001d6c',
              '#001141']
    Cyan10 = ['#e5f6ff', '#bae6ff', '#82cfff', '#33b1ff', '#1192e8', '#0072c3', '#00539a', '#003a6d', '#012749',
              '#1c0f30']
    Teal10 = ['#d9fbfb', '#9ef0f0', '#3ddbd9', '#08bdba', '#009d9a', '#007d79', '#005d5d', '#004144', '#022b30',
              '#081a1c']
    Orange10 = ['#fff2e6', '#ffd9b3', '#ffbf80', '#ffa64d', '#ff8c1a', '#e67300', '#b35900', '#804000', '#4d2600',
                '#1a0d00']


class DivergingPalette(Enum):
    # diverging colors with two hues for inherently ordered values
    # good for numeric variable has a meaningful central value, like zero. The colors gradients from the center
    # to the right and the left.
    # green and red theme
    Watermelon = ['#488f31', '#699f54', '#88af76', '#a5bf98', '#c2d0bc', '#dfdfdf', '#e3c2c1', '#e4a3a3', '#e18487',
                  '#dc646b', '#de425b']
    # blue and red theme
    Unclesam = ['#4589ff', '#779cfc', '#9baff9', '#b9c4f6', '#d4d8f2', '#dfdfdf', '#e3c2c1', '#e4a3a3', '#e18487',
                '#dc646b', '#de425b']
