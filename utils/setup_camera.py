import bpy
from math import pi

def rotate_camera(param):
    # add new dumb camera
    bpy.ops.object.camera_add()
    scene = bpy.data.scenes["Scene"]
    cam = bpy.data.objects['Camera']

    # rotation in Oxy
    scene.camera.rotation_euler[2] = param / 180 * pi
    cam.rotation_mode = 'XYZ'
    scene.camera = cam

    # delete dumb camera
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Camera.001'].select_set(True)
    bpy.ops.object.delete()
