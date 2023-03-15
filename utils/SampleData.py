from pydantic import BaseModel
from json import dumps, load


class Bbox(BaseModel):

    x: int
    y: int
    width: int
    height: int
    color: str
    type: str


class ImageBboxes(BaseModel):
    image_name: str
    bboxes: list[Bbox]


class SampleData(BaseModel):
    ImageBboxes: list[ImageBboxes]


def read_json(bboxes_file_path, already_rendered_num):
    with open(bboxes_file_path, "r+") as file:

        if already_rendered_num == 0:
            ls_im_bboxes = []
            file.close()
        else:
            datax = load(file)
            ls_im_bboxes = SampleData(**datax).ImageBboxes
            file.close()

    return ls_im_bboxes