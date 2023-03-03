from pydantic import BaseModel


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
