import os
# clear blender console
os.system('cls')
import sys
import time
import math
from random import random, randint, choice, choices, uniform
import bpy
import numpy as np
from json import dumps, load
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
from utils.setup_camera import rotate_camera
from utils.bboxes import camera_view_bounds_2d
from utils.get_edges import xxyyzz_edges
from utils.objects_in_view import select_objects_in_camera
from utils.make_grid import make_grid
from utils.SampleData import SampleData

def l2(obj):
    return round(np.sqrt((obj.location.x - bpy.data.objects['Camera'].location.x) ** 2 + (obj.location.y - bpy.data.objects['Camera'].location.y) ** 2 + (obj.location.z - bpy.data.objects['Camera'].location.z) ** 2), ndigits=0)

colors_list = {(0, 0.8, 0.8, 1): 'cian', (0.8, 0, 0.8, 1): 'purple', (0.8, 0.8, 0, 1): 'yellow', (0.1, 0, 0.8, 1): 'blue',
               (0, 0.8, 0, 1): 'green'}

shapes = {'round': 'circle', 'round2': 'circle', 'sqr': 'square', 'triu': 'triangle',
        'circle1': 'hoop', 'circle2': 'hoop', 'gate': 'gate', 'TARGET': 'Target circle'}

if not os.path.isdir('results'):
    os.mkdir('results')
    
if not os.path.isfile("bboxes.json"):
    Path("bboxes.json").touch()

model = pickle_load(open('resources/teknofest/SVM_mist_classifier_teknofest.sav', 'rb'))

if len(list(Path('./results').glob('*.jpg'))) == 0:
    num = 0
    
else:
    last_im = (sorted(Path('./results').glob('*.jpg'), key=os.path.getmtime)[-1]).name
    num = int(last_im.split('.')[0].split('_')[-1]) + 1

# angles of camera rotation
LEFT_ANGLE = 0
RIGHT_ANGLE = 360

# see readme
DRAW_BBOXES = True

RENDERING = True
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# images to render
NUM_IMAGES = 5

# for production necessary minimum 512
NUM_SAMPLES = 3

MIST = False

sc = jb_load('resources/teknofest/StandartScalerSVC_teknofest.bin')

if DRAW_BBOXES:
    # it's necessary to run another script to install cv2 to blender interpretator 
    import cv2
    
geometry_file_path = os.path.join(base_dir, "bboxes.json")
with open(geometry_file_path, "r+") as file:

    if num == 0:
        ls_im_bboxes = []
        file.close()
    else:
        datax = load(file)
        ls_im_bboxes = SampleData(**datax).ImageBboxes
        file.close()

with open(geometry_file_path, "r+") as file:
    objects_to_move = set(list(bpy.data.collections['move_and_change_color'].objects) + (list(bpy.data.collections['only_move'].objects)))
    # get low and high xxyy coordinates of bath
    low_x, low_y, low_z, high_x, high_y, high_z = xxyyzz_edges(bpy.data.objects['стена'])
    # save initial power of light
    medium_light = 100000
    init_lights = [bpy.data.objects['Источник-область'], bpy.data.objects['Источник-область.001']]
    # finding the biggest dimension to correctly relocate objects
    mx_dim = 0
    for object in objects_to_move:
        
        if object.dimensions.x > mx_dim:
            mx_dim = object.dimensions.x
            
        if object.dimensions.y > mx_dim:
             mx_dim = object.dimensions.y
              
    # object for hsv -> rgb transformation
    c = Color()
    
    for i in range(num, num + NUM_IMAGES):
    
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
                
                bpy.data.materials["Material.001"].node_tree.nodes["Principled Volume"].inputs[2].default_value = 0.05
            
        
        # get grid for objects
        grid = make_grid(len(objects_to_move), low_x, low_y, high_x, high_y, math.ceil(mx_dim))
        
        # rotating and changing colors
        for j, object in enumerate(objects_to_move):
            
            # x and y are swapped in blender
            object.location.x = low_x + grid[j][1]
            object.location.y = low_y + grid[j][0]
            
            if object in list(bpy.data.collections['move_and_change_color'].objects):
                color = choice(list(colors_list.keys()))
                # rotate only in Oxy
                object.rotation_euler[2] = randint(0, 360) / 180 * math.pi
                object.active_material.node_tree.nodes['Principled BSDF'].inputs["Base Color"].default_value = color
            else:
                object.rotation_euler[2] = randint(0, 360) / 180 * math.pi
                 
        # randomize coordinates for camera    
        x_coords = list(range(low_x, high_x))
        y_coords = list(range(low_y, high_y))
        
        # rotating camera until we get > 2 objects in view
        while 1:
            
            # relocate camera
            bpy.data.objects['Camera'].location.x = choice(x_coords)
            bpy.data.objects['Camera'].location.y = choice(y_coords)
            bpy.data.objects['Camera'].location.z = uniform(low_z, high_z)  
        
            rotate_camera(uniform(LEFT_ANGLE, RIGHT_ANGLE))
            
            # there are a lot of objects thats we don't need in ROI, i.e pools walls, water, Sun etc
            objects_in_camera = objects_to_move.intersection(select_objects_in_camera())
            valid_objects = []
            
            # check how much objects in camera
            if len(objects_in_camera) < 3:
                continue

            visible_obj = []
            if MIST:
                # check visibility
                visible = 0
                for object in objects_in_camera:
                    temp = sc.transform(np.array([density, l2(object)]).reshape(1, -1))
                    
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
            path = os.path.join(base_dir, 'results', f)
            bpy.context.scene.render.filepath = path
            bpy.context.scene.render.resolution_x = IMAGE_WIDTH
            bpy.context.scene.render.resolution_y = IMAGE_HEIGHT
            bpy.context.scene.render.image_settings.file_format='JPEG'
            bpy.ops.render.render(write_still = True)
            bpy.context.scene.cycles.samples = NUM_SAMPLES
            if DRAW_BBOXES:
                im = cv2.imread(path)
                for box in valid_objects:
                    cv2.rectangle(im, (box.x, box.y), (box.x + box.width, box.y + box.height), (255, 0, 0), 1)
                    cv2.putText(im, box.name, (box.x, box.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)
                    cv2.putText(im, f'{str(box.width)}, {str(box.height)}', (box.x, box.y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)
                os.remove(path)
                path = os.path.join(base_dir, 'results', f'image_{i}.jpg')
                cv2.imwrite(path, im)
                
    m = SampleData(ImageBboxes = ls_im_bboxes)
    st = m.json(indent=2)
    file.write(st)