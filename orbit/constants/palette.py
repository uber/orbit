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