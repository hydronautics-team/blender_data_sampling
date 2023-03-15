import os
import cv2
import bpy


def render(rendering_path, f, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_SAMPLES, DRAW_BBOXES, valid_objects, i):
    path = os.path.join(rendering_path, f)
    bpy.context.scene.render.filepath = path
    bpy.context.scene.render.resolution_x = IMAGE_WIDTH
    bpy.context.scene.render.resolution_y = IMAGE_HEIGHT
    bpy.context.scene.render.image_settings.file_format = 'JPEG'
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.cycles.samples = NUM_SAMPLES

    if DRAW_BBOXES:
        im = cv2.imread(path)
        for box in valid_objects:
            cv2.rectangle(im, (box.x, box.y), (box.x + box.width, box.y + box.height), (255, 0, 0), 1)
            cv2.putText(im, box.name, (box.x, box.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)
            cv2.putText(im, f'{str(box.width)}, {str(box.height)}', (box.x, box.y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

        os.remove(path)
        path = os.path.join(rendering_path, f'image_{i}.jpg')
        cv2.imwrite(path, im)
