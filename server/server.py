from fastapi import FastAPI, File, UploadFile
import numpy as np
import edge_detection
import io
from PIL import Image


app: FastAPI = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/floorplan")
async def read_floorplan(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    contours: np.ndarray = edge_detection.get_contours(image)
    return {"contours": contours.tolist()}
