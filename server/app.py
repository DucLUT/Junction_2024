"""
This module provides an API for processing floorplan images.
"""

import io
import logging
import subprocess
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from PIL import Image
import pdf2image
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app: FastAPI = FastAPI()


@app.get("/")
async def root():
    """
    Root endpoint that returns a simple message.
    """
    return {"message": "Hello World"}


@app.post("/api/floorplan")
async def read_floorplan(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to process a floorplan image or PDF.

    Args:
        request (Request): The request object.
        file (UploadFile): The uploaded file.

    Returns:
        dict: A dictionary containing the wall contours.
    """
    logger.info("Request: %s", request)
    logger.info("Headers: %s", request.headers)
    logger.info("Received file: %s", file.filename)

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    contents = await file.read()
    image = None
    if file.content_type == "application/pdf":
        logger.info("Converting PDF to image")
        # utils.read_pdf_layers(contents)
        # there is no utils pdf to image function
        image = pdf2image.convert_from_bytes(contents)[0]
    else:
        try:
            logger.info("Reading image file")
            image = Image.open(io.BytesIO(contents))
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid image file: {exc}"
            ) from exc

    if image is None:
        raise HTTPException(status_code=400, detail="No valid image found")

    try:
        wall_contours = utils.extract_walls(image)
        contours_list = [
            contour.tolist() for contour in wall_contours
        ]  # Convert to list
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {exc}"
        ) from exc

    return {"contours": contours_list}
