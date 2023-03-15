from math import ceil, floor


def xxyyzz_edges(obj):

    low_x = obj.location.x - obj.dimensions.x / 2
    high_x = obj.location.x + obj.dimensions.x / 2

    low_y = obj.location.y - obj.dimensions.y / 2
    high_y = obj.location.y + obj.dimensions.y / 2

    # obj.location.z is low edge of object
    low_z = obj.location.z + 0.55
    high_z = obj.location.z + obj.dimensions.z - 0.55

    # out of harm's way
    return ceil(low_x) + 3, ceil(low_y) + 3, low_z, floor(high_x) - 3, floor(high_y) - 3, high_z


def find_max_dim(objects):
    mx_dim = 0
    for object in objects:
        if object.dimensions.x > mx_dim:
            mx_dim = object.dimensions.x
        if object.dimensions.y > mx_dim:
            mx_dim = object.dimensions.y

    return mx_dim
