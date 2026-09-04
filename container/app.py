"""Grand Challenge 'invoke' HTTP server for the DoseRAD2026 algorithm container.

Contract (from GC's algorithm template / DoseRAD submission instructions):
  * Listen on 0.0.0.0:4743  (port is hardcoded by the platform — do not change).
  * GET  /health  -> 200 once the model is loaded; 503 while still loading.
                     Must NOT redirect (a 302 is treated as failure).
  * POST /invoke  -> process whatever is mounted at /input, write /output,
                     respond 201 on success.
  * Model is loaded once at startup (unmetered), never inside /invoke.
The image must carry:  LABEL org.grand-challenge.api-method="invoke"

Inference logic lives in inference.py (init_model / run) — this file is just the
server shell and should not need editing.
"""

from __future__ import annotations

import contextlib
import traceback

import uvicorn
from fastapi import FastAPI, Response

import inference

PORT = 4743  # hardcoded by Grand Challenge
_STATE: dict[str, object] = {"model": None, "ready": False}


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the model before we ever report healthy. This time is free
    # (the platform does not measure it against the /invoke budget).
    _STATE["model"] = inference.init_model()
    _STATE["ready"] = True
    yield
    _STATE["ready"] = False
    _STATE["model"] = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> Response:
    # Plain 200/503 — never a redirect.
    return Response(status_code=200 if _STATE["ready"] else 503)


@app.post("/invoke")
def invoke() -> Response:
    if not _STATE["ready"] or _STATE["model"] is None:
        return Response(status_code=503)
    try:
        inference.run(_STATE["model"])
    except Exception:  # surface the traceback in the (preliminary-phase) logs
        traceback.print_exc()
        return Response(status_code=500)
    return Response(status_code=201)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
