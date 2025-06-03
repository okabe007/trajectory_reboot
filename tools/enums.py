from enum import Enum

class IOStatus(Enum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    SURFACE = "surface"
    REFLECT = "reflect"
    STICK = "stick"
    BORDER = "border"
    POLYGON_MODE = "polygon_mode"
    BOTTOM_OUT = "bottom_out"
    SPOT_EDGE_OUT = "spot_edge_out"

class SpotIO(Enum):
    INSIDE = "inside"
    REFLECT = "reflect"
    STICK = "stick"
    BORDER = "border"
    POLYGON_MODE = "polygon_mode"
    BOTTOM_OUT = "bottom_out"
    SPHERE_OUT = "sphere_out"
