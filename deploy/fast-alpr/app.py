import os
from statistics import mean
from typing import Any

import cv2
import numpy as np
from fast_alpr import ALPR
from fastapi import FastAPI, File, HTTPException, UploadFile


def _confidence_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, list):
        return float(mean(value)) if value else 0.0
    return float(value)


def _serialize_result(result: Any) -> dict[str, Any]:
    detection = result.detection
    bbox = detection.bounding_box
    ocr = result.ocr
    return {
        "plate": ocr.text if ocr and ocr.text else None,
        "confidence": _confidence_value(ocr.confidence if ocr else None),
        "region": ocr.region if ocr else None,
        "region_confidence": float(ocr.region_confidence) if ocr and ocr.region_confidence else None,
        "detection_confidence": float(detection.confidence),
        "bounding_box": {
            "x1": int(bbox.x1),
            "y1": int(bbox.y1),
            "x2": int(bbox.x2),
            "y2": int(bbox.y2),
        },
    }


app = FastAPI(title="fast-alpr-container")
alpr = ALPR(
    detector_conf_thresh=float(os.getenv("FAST_ALPR_DETECTOR_CONF", "0.4")),
    ocr_device=os.getenv("FAST_ALPR_OCR_DEVICE", "cpu"),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)) -> dict[str, Any]:
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty image")

    array = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="invalid image")

    results = alpr.predict(frame)
    serialized = [_serialize_result(result) for result in results]
    return {"results": serialized, "count": len(serialized)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8090")))
