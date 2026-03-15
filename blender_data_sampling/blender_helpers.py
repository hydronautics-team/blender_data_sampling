from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

import bpy
from mathutils import Color, Vector


@dataclass(slots=True)
class ViewBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    object_name: str
    dim_x: float
    dim_y: float

    @property
    def x(self) -> int:
        return round(self.min_x * self.dim_x)

    @property
    def y(self) -> int:
        return round(self.dim_y - self.max_y * self.dim_y)

    @property
    def width(self) -> int:
        return round((self.max_x - self.min_x) * self.dim_x)

    @property
    def height(self) -> int:
        return round((self.max_y - self.min_y) * self.dim_y)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def camera_view_bounds_2d(scene, camera_object, mesh_object) -> ViewBox | None:
    matrix = camera_object.matrix_world.normalized().inverted()
    mesh = mesh_object.to_mesh(preserve_all_data_layers=True)
    try:
        mesh.transform(mesh_object.matrix_world)
        mesh.transform(matrix)

        camera = camera_object.data
        frame_template = [-vertex for vertex in camera.view_frame(scene=scene)[:3]]
        is_perspective = camera.type != "ORTHO"

        xs: list[float] = []
        ys: list[float] = []

        visible_vertex_count = 0
        for vertex in mesh.vertices:
            local = vertex.co
            depth = -local.z
            frame = frame_template

            if is_perspective:
                if depth <= 0.0:
                    continue
                frame = [(frame_vertex / (frame_vertex.z / depth)) for frame_vertex in frame_template]

            visible_vertex_count += 1

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            xs.append((local.x - min_x) / (max_x - min_x))
            ys.append((local.y - min_y) / (max_y - min_y))

        if visible_vertex_count == 0 or not xs or not ys:
            return None

        min_x = clamp(min(xs), 0.0, 1.0)
        max_x = clamp(max(xs), 0.0, 1.0)
        min_y = clamp(min(ys), 0.0, 1.0)
        max_y = clamp(max(ys), 0.0, 1.0)

        if max_x <= min_x or max_y <= min_y:
            return None

        render = scene.render
        factor = render.resolution_percentage * 0.01
        return ViewBox(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            object_name=mesh_object.name,
            dim_x=render.resolution_x * factor,
            dim_y=render.resolution_y * factor,
        )
    finally:
        mesh_object.to_mesh_clear()


def camera_as_planes(scene, camera_object):
    from mathutils.geometry import normal

    camera = camera_object.data
    matrix = camera_object.matrix_world.normalized()
    frame = [matrix @ vertex for vertex in camera.view_frame(scene=scene)]
    origin = matrix.to_translation()

    planes = []
    is_perspective = camera.type != "ORTHO"
    for index in range(4):
        frame_other = origin if is_perspective else frame[index] + matrix.col[2].xyz
        normal_vector = normal(frame_other, frame[index - 1], frame[index])
        distance = -normal_vector.dot(frame_other)
        planes.append((normal_vector, distance))

    if not is_perspective:
        normal_vector = normal(frame[0], frame[1], frame[2])
        planes.append((normal_vector, -normal_vector.dot(origin)))

    return planes


def side_of_plane(plane, vector) -> float:
    return plane[0].dot(vector) + plane[1]


def is_segment_in_planes(point1, point2, planes) -> bool:
    delta = point2 - point1
    point1_factor = 0.0
    point2_factor = 1.0

    for plane in planes:
        divisor = delta.dot(plane[0])
        if divisor == 0.0:
            continue

        numerator = -side_of_plane(plane, point1)
        if divisor > 0.0:
            if numerator >= divisor:
                return False
            if numerator > 0.0:
                factor = numerator / divisor
                point1_factor = max(factor, point1_factor)
                if point1_factor > point2_factor:
                    return False
        else:
            if numerator > 0.0:
                return False
            if numerator > divisor:
                factor = numerator / divisor
                point2_factor = min(factor, point2_factor)
                if point1_factor > point2_factor:
                    return False
    return True


def point_in_object(obj, point) -> bool:
    xs = [vertex[0] for vertex in obj.bound_box]
    ys = [vertex[1] for vertex in obj.bound_box]
    zs = [vertex[2] for vertex in obj.bound_box]
    local = obj.matrix_world.inverted() @ point
    return min(xs) <= local.x <= max(xs) and min(ys) <= local.y <= max(ys) and min(zs) <= local.z <= max(zs)


def object_in_planes(obj, planes) -> bool:
    box = [obj.matrix_world @ Vector(vertex) for vertex in obj.bound_box]
    if any(all(side_of_plane(plane, vertex) > 0.0 for plane in planes) for vertex in box):
        return True

    edges = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 7),
        (5, 6),
        (6, 7),
    )
    return any(is_segment_in_planes(box[start], box[end], planes) for start, end in edges)


def select_objects_in_camera(scene, camera_object) -> set[str]:
    origin = camera_object.matrix_world.to_translation()
    planes = camera_as_planes(scene, camera_object)
    return {
        obj.name
        for obj in scene.objects
        if point_in_object(obj, origin) or object_in_planes(obj, planes)
    }


def calculate_distance(source_object, target_object) -> float:
    dx = target_object.location.x - source_object.location.x
    dy = target_object.location.y - source_object.location.y
    dz = target_object.location.z - source_object.location.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def bounds_from_object(obj, xy_padding: int, z_padding: float) -> tuple[int, int, float, int, int, float]:
    low_x = math.ceil(obj.location.x - obj.dimensions.x / 2) + xy_padding
    high_x = math.floor(obj.location.x + obj.dimensions.x / 2) - xy_padding
    low_y = math.ceil(obj.location.y - obj.dimensions.y / 2) + xy_padding
    high_y = math.floor(obj.location.y + obj.dimensions.y / 2) - xy_padding
    low_z = obj.location.z + z_padding
    high_z = obj.location.z + obj.dimensions.z - z_padding
    return low_x, low_y, low_z, high_x, high_y, high_z


def find_max_dimension(objects: Iterable) -> float:
    maximum = 0.0
    for obj in objects:
        maximum = max(maximum, obj.dimensions.x, obj.dimensions.y)
    return maximum


def generate_grid_positions(
    count: int,
    x_low: int,
    y_low: int,
    x_high: int,
    y_high: int,
    max_dim: int,
    rng: Random,
) -> list[tuple[int, int]]:
    grid = [
        [1 for _ in range(x_low + max_dim, x_high - max_dim + 1)]
        for _ in range(y_low + max_dim, y_high - max_dim + 1)
    ]
    if not grid or not grid[0]:
        raise ValueError("Sampling grid is empty. Check bounds or paddings.")

    positions: list[tuple[int, int]] = []
    for _ in range(count):
        attempts = 0
        while True:
            x_index = rng.randint(0, len(grid) - 1)
            y_index = rng.randint(0, len(grid[0]) - 1)
            attempts += 1
            if grid[x_index][y_index] != 0:
                break
            if attempts > 10_000:
                raise RuntimeError("Unable to place all objects on the sampling grid")

        grid[x_index][y_index] = 0
        positions.append((x_index, y_index))

        for row in range(max(0, x_index - max_dim), min(len(grid), x_index + max_dim + 1)):
            for column in range(max(0, y_index - max_dim), min(len(grid[0]), y_index + max_dim + 1)):
                grid[row][column] = 0
    return positions


def rotate_camera(camera_object, angle_deg: float) -> None:
    camera_object.rotation_mode = "XYZ"
    camera_object.rotation_euler[2] = math.radians(angle_deg)


def is_object_in_front_of_camera(camera_object, target_object) -> bool:
    relative = camera_object.matrix_world.inverted() @ target_object.matrix_world.to_translation()
    return relative.z < 0.0


def aim_camera_at_point(
    camera_object,
    target_point,
    base_rotation_euler=None,
    yaw_jitter_deg: float = 0.0,
    pitch_jitter_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> None:
    direction = target_point - camera_object.matrix_world.to_translation()
    if direction.length == 0:
        return

    look_rotation = direction.to_track_quat("-Z", "Y").to_euler()
    if base_rotation_euler is None:
        rotation = look_rotation
        rotation.x += math.radians(pitch_jitter_deg)
        rotation.y += math.radians(roll_deg)
    else:
        rotation = base_rotation_euler.copy()
        rotation.x += math.radians(pitch_jitter_deg)
        rotation.y += math.radians(roll_deg)
        rotation.z = look_rotation.z
    rotation.z += math.radians(yaw_jitter_deg)
    camera_object.rotation_mode = "XYZ"
    camera_object.rotation_euler = rotation


def place_camera_near_target(
    camera_object,
    target_object,
    bounds: tuple[int, int, float, int, int, float],
    base_rotation_euler,
    horizontal_distance_range: tuple[float, float],
    vertical_offset_range: tuple[float, float],
    yaw_jitter_deg_range: tuple[float, float],
    pitch_jitter_deg_range: tuple[float, float],
    rng: Random,
) -> None:
    low_x, low_y, low_z, high_x, high_y, high_z = bounds
    target_location = target_object.matrix_world.to_translation()

    for _ in range(100):
        horizontal_distance = rng.uniform(*horizontal_distance_range)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        candidate_x = target_location.x + math.cos(angle) * horizontal_distance
        candidate_y = target_location.y + math.sin(angle) * horizontal_distance
        candidate_z = target_location.z + rng.uniform(*vertical_offset_range)
        if low_x <= candidate_x <= high_x and low_y <= candidate_y <= high_y and low_z <= candidate_z <= high_z:
            break
    else:
        candidate_x = clamp(target_location.x, low_x, high_x)
        candidate_y = clamp(target_location.y, low_y, high_y)
        candidate_z = clamp(target_location.z + vertical_offset_range[1], low_z, high_z)

    camera_object.location.x = candidate_x
    camera_object.location.y = candidate_y
    camera_object.location.z = candidate_z
    aim_camera_at_point(
        camera_object,
        target_location,
        base_rotation_euler=base_rotation_euler,
        yaw_jitter_deg=rng.uniform(*yaw_jitter_deg_range),
        pitch_jitter_deg=rng.uniform(*pitch_jitter_deg_range),
    )


def set_object_color_from_hsv(object_name: str, hsv: dict[str, tuple[float, float]], rng: Random) -> None:
    color = Color()
    color.hsv = (
        rng.uniform(*hsv["h"]),
        rng.uniform(*hsv["s"]),
        rng.uniform(*hsv["v"]),
    )
    bpy.data.objects[object_name].color = (color.r, color.g, color.b, 1.0)


def _resolve_node_target(spec: dict):
    if spec.get("object_name"):
        return bpy.data.objects[spec["object_name"]].active_material.node_tree.nodes[spec["node_name"]]
    return bpy.data.materials[spec["material_name"]].node_tree.nodes[spec["node_name"]]


def set_material_scalar_input(spec: dict, input_key: str | int, value: float) -> None:
    node = _resolve_node_target(spec)
    node.inputs[input_key].default_value = value


def apply_surface_hsv_adjustment(spec: dict, rng: Random) -> None:
    set_material_scalar_input(spec, spec["hue_input"], rng.uniform(*spec["hue_range"]))
    set_material_scalar_input(spec, spec["saturation_input"], rng.uniform(*spec["saturation_range"]))
    set_material_scalar_input(spec, spec["value_input"], rng.uniform(*spec["value_range"]))


def set_material_color_from_hsv(spec: dict, input_key: str | int, hsv: dict[str, tuple[float, float]], rng: Random) -> None:
    color = Color()
    color.hsv = (
        rng.uniform(*hsv["h"]),
        rng.uniform(*hsv["s"]),
        rng.uniform(*hsv["v"]),
    )
    node = _resolve_node_target(spec)
    node.inputs[input_key].default_value = (color.r, color.g, color.b, 1.0)


def relocate_objects(
    objects,
    positions: list[tuple[int, int]],
    low_x: int,
    low_y: int,
    recolor_collection: str | None,
    color_palette: list[tuple[float, float, float, float]],
    material_node_name: str,
    material_input_key: str | int,
    rng: Random,
) -> None:
    recolor_names = set()
    if recolor_collection:
        recolor_names = {obj.name for obj in bpy.data.collections[recolor_collection].objects}

    for obj, position in zip(objects, positions):
        obj.location.x = low_x + position[1]
        obj.location.y = low_y + position[0]
        obj.rotation_euler[2] = math.radians(rng.uniform(0.0, 360.0))

        if obj.name in recolor_names and color_palette:
            node = obj.active_material.node_tree.nodes[material_node_name]
            node.inputs[material_input_key].default_value = rng.choice(color_palette)


def render_image(output_path: Path, width: int, height: int, samples: int) -> None:
    bpy.context.scene.render.filepath = str(output_path)
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.image_settings.file_format = "JPEG"
    bpy.context.scene.cycles.samples = samples
    bpy.ops.render.render(write_still=True)


def update_scene() -> None:
    bpy.context.view_layer.update()
