from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import io
import logging
from PIL import Image
import cv2
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/floorplan")
async def read_floorplan(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to process a floorplan image or PDF.

    Args:
        request (Request): The request object.
        file (UploadFile): The uploaded file.

    Returns:
        dict: A dictionary containing the wall contours and line coordinates.
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
        # Preprocess the image
        preprocessed_image = utils.preprocess_image(image)

        # Process the contours
        processed_image = utils.process_contours(preprocessed_image)

        # Save the processed image to a file
        processed_image_path = "processed_image.jpg"
        cv2.imwrite(processed_image_path, processed_image)
        logger.info(f"Processed image saved to {processed_image_path}")

        # Extract wall contours
       
       

        # Extract straight lines and their coordinates
        line_image, line_coordinates = utils.keep_straight_lines(preprocessed_image)

        # Save the image with lines
        line_image_path = "line_image.jpg"
        cv2.imwrite(line_image_path, line_image)
        logger.info(f"Line image saved to {line_image_path}")

        # Convert numpy.int32 to native Python int
        line_coordinates = [
            (int(x1), int(y1), int(x2), int(y2)) for (x1, y1, x2, y2) in line_coordinates
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {exc}"
        ) from exc

    return {"lines": line_coordinates}