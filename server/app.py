from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import utils
import io
from PIL import Image
from pdf2image import convert_from_bytes
from typing import List

app: FastAPI = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/api/floorplan")
async def read_floorplan(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    contents = await file.read()
    if file.content_type == "application/pdf":
        images = convert_from_bytes(contents)
        image = images[0]
    else:
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        contours: List[np.ndarray] = utils.get_wall_contours(image)
        contours_list = [contour.tolist() for contour in contours]  # Convert to list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    return {"contours": contours_list}
