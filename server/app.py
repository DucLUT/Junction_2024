from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import numpy as np
import utils
import io
from PIL import Image
from pdf2image import convert_from_bytes
from typing import List
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app: FastAPI = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/api/floorplan")
async def read_floorplan(request: Request, file: UploadFile = File(...)):

    logger.info("Request: %s", request)
    logger.info("Headers: %s", request.headers)
    logger.info("Received file: %s", file.filename)

    # Check if Tesseract is installed
    try:
        subprocess.run(
            ["tesseract", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        raise HTTPException(
            status_code=500,
            detail="Tesseract is not installed or it's not in your PATH. See README file for more information.",
        )

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    contents = await file.read()
    if file.content_type == "application/pdf":
        logger.info("Converting PDF to image")
        images = convert_from_bytes(contents)
        image = images[0]
    else:
        try:
            logger.info("Reading image file")
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        wall_contours = utils.extract_walls(image)
        contours_list = [
            contour.tolist() for contour in wall_contours
        ]  # Convert to list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    return {"contours": contours_list}
