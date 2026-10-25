import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from inference import Inspector

app = FastAPI(title="PCB Inspector", version="2.0")
inspector = Inspector()


async def read_inputs(image, reference_id):
    data = await image.read()
    board = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if board is None:
        raise HTTPException(status_code=400, detail="could not read image")
    ref = inspector.load_reference(reference_id) if reference_id else None
    return board, ref


def run(board, ref):
    try:
        return inspector.inspect(board, ref)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "device": inspector.device}


@app.post("/inspect")
async def inspect(image: UploadFile = File(...), reference_id: str | None = Form(None)):
    board, ref = await read_inputs(image, reference_id)
    return run(board, ref)


@app.post("/inspect/image")
async def inspect_image(image: UploadFile = File(...), reference_id: str | None = Form(None)):
    board, ref = await read_inputs(image, reference_id)
    ok, png = cv2.imencode(".png", inspector.annotate(board, run(board, ref)))
    return Response(content=png.tobytes(), media_type="image/png")