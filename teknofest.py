import os
# clear blender console
os.system('cls')
import sys

import subprocess
sys.path.append(subprocess.check_output(['pipenv', '--venv']).strip().decode('utf-8') + '/lib/python3.10/site-packages')

import math
from random import random, randint, choice, choices, uniform
import bpy
import yaml
import numpy as np
from mathutils import Color
from pathlib import Path
from pickle import load as pickle_load
from joblib import load as jb_load

# insert your's pathes
if os.name == 'posix':
    base_dir = '/home/alex/python_projects/blender_data_sampling'
elif os.name == 'nt':
    base_dir = 'e:\\blender_data_sampling'

os.chdir(base_dir)
sys.path.append(base_dir)

NAME = (__file__.split('/')[-1]).split('.')[0]
stream = open(f'config/{NAME}/params.yaml', 'r')
config = yaml.load(stream, yaml.Loader)

from utils.setup_camera import rotate_camera
from utils.bboxes import camera_view_bounds_2d
from utils.get_edges import xxyyzz_edges, find_max_dim
from utils.objects_in_view import select_objects_in_camera, l2, relocate_objects
from utils.make_grid import make_grid
from utils.SampleData import SampleData, read_json
from utils.rendering import render

colors_list = config['colors']
shapes = config['shapes']

DRAW_BBOXES = config['render_settings']['DRAW_BBOXES']
RENDERING = config['render_settings']['RENDERING']
IMAGE_WIDTH = config['render_settings']['IMAGE_WIDTH']
IMAGE_HEIGHT = config['render_settings']['IMAGE_HEIGHT']

# images to render
NUM_IMAGES = config['render_settings']['NUM_IMAGES']

# render quality
# for production necessary minimum 512
NUM_SAMPLES = config['render_settings']['NUM_SAMPLES']

MIST = config['render_settings']['MIST']

rendering_path = f'results/{NAME}'
json_name = f'{NAME}_bboxes.json'
bboxes_file_path = os.path.join(base_dir, json_name)

if not os.path.isdir(rendering_path):
    os.mkdir(rendering_path)
    
if not os.path.isfile(json_name):
    Path(json_name).touch()

# SVM classifier to define visible / not visible
model = pickle_load(open(f'resources/{NAME}/SVM_mist_classifier_teknofest.sav', 'rb'))
# scaler for input features for SVM
scaler = jb_load(f'resources/{NAME}/StandartScalerSVC_teknofest.bin')

if len(list(Path(f'results/{NAME}').glob('*.jpg'))) == 0:
    already_rendered_num = 0
    
else:
    last_im = (sorted(Path(rendering_path).glob('*.jpg'), key=os.path.getmtime)[-1]).name
    already_rendered_num = int(last_im.split('.')[0].split('_')[-1]) + 1

ls_im_bboxes = read_json(bboxes_file_path, already_rendered_num)

with open(bboxes_file_path, "r+") as file:

    objects_to_move = set(list(bpy.data.collections['move_and_change_color'].objects) + (list(bpy.data.collections['only_move'].objects)))

    # get low and high xxyy coordinates of bath
    low_x, low_y, low_z, high_x, high_y, high_z = xxyyzz_edges(bpy.data.objects['стена'])

    # save initial power of light
    medium_light = 100000
    init_lights = [bpy.data.objects['Источник-область'], bpy.data.objects['Источник-область.001']]

    # finding the biggest dimension to correctly relocate objects
    # to avoid intersection of object and bath wall
    mx_dim = find_max_dim(objects_to_move)

    # object for hsv -> rgb transformation
    c = Color()
    
    for i in range(already_rendered_num, already_rendered_num + NUM_IMAGES):
    
        h = uniform(0.5, 0.6)
        s = uniform(0.6, 0.9)
        v = uniform(0.5, 0.7)
    
        # change walls' color
        c.hsv = h, s, v
        bpy.data.objects['стена'].color = c.r, c.g, c.b, 1.0
         
        # change color of baths' floor
        bpy.data.objects['стена'].active_material.node_tree.nodes["Hue Saturation Value"].inputs[0].default_value = uniform(0.45, 0.53)
        bpy.data.objects['стена'].active_material.node_tree.nodes["Hue Saturation Value"].inputs[1].default_value = uniform(1.1, 2)
        bpy.data.objects['стена'].active_material.node_tree.nodes["Hue Saturation Value"].inputs[2].default_value = uniform(0.9, 2)

        if MIST:

            density = uniform(0, 0.45)
            bpy.data.materials["Material.001"].node_tree.nodes["Principled Volume"].inputs[2].default_value = density
            
            # water HSV
            c.hsv = uniform(0.5, 0.6), uniform(0.65, 0.8), uniform(0.15, 0.35)
            bpy.data.materials["Material.001"].node_tree.nodes["Principled Volume"].inputs[0].default_value = (c.r, c.g, c.b, 1)

            for j, lt in enumerate(list(bpy.data.collections['lights'].objects)):
                lt.location.x = init_lights[j].location.x
                lt.location.y = init_lights[j].location.y
                lt.data.energy = medium_light
                lt.rotation_euler[2] = init_lights[j].rotation_euler[2]

        else:
            for light in list(bpy.data.collections['lights'].objects):
                # relocate and change power of light
                light.location.x = uniform(low_x, high_x)
                light.location.y = uniform(low_y, high_y)
                light.data.energy = uniform(medium_light - 0.25 * medium_light, medium_light + 0.75 * medium_light)
                light.rotation_euler[2] = randint(0, 360) / 180 * math.pi
                # mist density
                bpy.data.materials["Material.001"].node_tree.nodes["Principled Volume"].inputs[2].default_value = 0.05

        # get grid of locations for objects
        grid = make_grid(len(objects_to_move), low_x, low_y, high_x, high_y, math.ceil(mx_dim))

        # rotating and changing colors
        relocate_objects(objects_to_move, low_x, low_y, grid, colors_list)

        # randomize coordinates for camera    
        x_coords = list(range(low_x, high_x))
        y_coords = list(range(low_y, high_y))
        
        # rotating camera until we get > 2 objects in view
        while 1:
            
            # relocate camera
            bpy.data.objects['Camera'].location.x = choice(x_coords)
            bpy.data.objects['Camera'].location.y = choice(y_coords)
            bpy.data.objects['Camera'].location.z = uniform(low_z, high_z)  
        
            rotate_camera(uniform(0, 360))
            
            # there are a lot of objects thats we don't need in ROI, i.e pools walls, water, Sun etc
            objects_in_camera = objects_to_move.intersection(select_objects_in_camera())
            valid_objects = []
            
            # check how much camera sees
            if len(objects_in_camera) < 3:
                continue

            if MIST:
                visible_obj = []
                # check visibility
                visible = 0
                for object in objects_in_camera:
                    temp = scaler.transform(np.array([density, l2(object)]).reshape(1, -1))

                    if model.predict(temp)[0] == 1:
                        visible += 1
                        visible_obj.append(object)

                if visible < 2:
                    continue
                
            else:
                visible_obj = objects_in_camera

            # checking invalid bboxes
            for object in visible_obj:

                # get bbox of object
                b = camera_view_bounds_2d(bpy.context.scene, bpy.context.scene.camera, bpy.data.objects[object.name])
            
                # check size of bbox; in far figures may looks like a string
                if b.width < 30 or b.height < 5:
                    continue
                
                # check distance between camera and object
                elif np.sqrt((object.location.x - bpy.data.objects['Camera'].location.x) ** 2 + (object.location.y - bpy.data.objects['Camera'].location.y) ** 2 + (object.location.z - bpy.data.objects['Camera'].location.z) ** 2) < 1.5:
                    continue
                
                else:
                    valid_objects.append(b)     

            # saving and rendering
            if len(valid_objects) >= 2:
                list_bboxes = []
                f = "image_" + str(i) + ".jpg"
                for obj in valid_objects:
                    if obj.color == (0.8, 0.0, 0.0, 1.0):
                        list_bboxes.append({'x': obj.x, 'y': obj.y, 'width': obj.width, 'height': obj.height,
                                            'color': 'red', 'type': shapes[obj.name]})

                    elif obj.color == (0.0, 0.0, 0.0, 1.0):
                        list_bboxes.append({'x': obj.x, 'y': obj.y, 'width': obj.width, 'height': obj.height,
                                            'color': 'black', 'type': shapes[obj.name]})

                    else:
                        list_bboxes.append({'x': obj.x, 'y': obj.y, 'width': obj.width, 'height': obj.height,
                                            'color': colors_list[obj.color], 'type': shapes[obj.name]})

                im_bbox = {'image_name': f, 'bboxes': list_bboxes}
                ls_im_bboxes.append(im_bbox)
                break

        if RENDERING:        
            render(rendering_path, f, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_SAMPLES, DRAW_BBOXES, valid_objects, i)
                
    m = SampleData(ImageBboxes=ls_im_bboxes)
    st = m.json(indent=2)
    file.write(st)
