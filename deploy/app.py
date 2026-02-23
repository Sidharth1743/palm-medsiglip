from __future__ import annotations

import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from palm_deploy.predictor import PalmAnemiaPredictor


app = FastAPI()
predictor: PalmAnemiaPredictor | None = None


@app.on_event("startup")
def _load_model() -> None:
    global predictor
    predictor = PalmAnemiaPredictor(os.environ.get("ARTIFACT_DIR"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    image = Image.open(file.file)
    result = predictor.predict(image) if predictor is not None else {"error": "model_not_loaded"}
    return JSONResponse(content=result)
