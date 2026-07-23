
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from ...utils import logging
from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from .chat_completion import ChatCompletionHandler
    from .completion import CompletionHandler
    from .response import ResponseHandler
    from .transcription import TranscriptionHandler

from .model_manager import ModelManager
from .utils import X_REQUEST_ID, CBWorkerDeadError, GenerationState


logger = logging.get_logger(__name__)


def build_server(
    model_manager: ModelManager,
    chat_handler: "ChatCompletionHandler",
    completion_handler: "CompletionHandler",
    response_handler: "ResponseHandler",
    transcription_handler: "TranscriptionHandler",
    generation_state: GenerationState,
    enable_cors: bool = False,
) -> "FastAPI":
    """Build and return a configured FastAPI application.

    Args:
        model_manager: Handles model loading, caching, and cleanup.
        chat_handler: Handles `/v1/chat/completions` requests.
        response_handler: Handles `/v1/responses` requests.
        generation_state: Owns the per-model generation managers (regular and CB). Passed
            in here so `/health` can check whether the CB worker has died and respond with
            503 instead of a misleading 200.
        enable_cors: If `True`, adds permissive CORS middleware (allow all origins).

    Returns:
        A FastAPI app ready to be passed to uvicorn.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pass

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(CBWorkerDeadError)
    async def _cb_dead_handler(_request: Request, exc: CBWorkerDeadError):
        pass

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.warning_once("CORS allow origin is set to `*`. Not recommended for production.")


    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        pass


    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: dict):
        pass

    @app.post("/v1/completions")
    async def completions(request: Request, body: dict):
        pass

    @app.post("/v1/responses")
    async def responses(request: Request, body: dict):
        pass

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(request: Request):
        pass

    @app.post("/load_model")
    async def load_model(body: dict):
        from fastapi import HTTPException

        model = body.get("model")
        if model is None:
            raise HTTPException(status_code=422, detail="Missing `model` field in the request body.")
        model_id_and_revision = model_manager.process_model_name(model)
        return StreamingResponse(
            model_manager.load_model_streaming(model_id_and_revision), media_type="text/event-stream"
        )

    @app.post("/reset")
    def reset():
        model_manager.shutdown()
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    @app.options("/v1/models")
    def list_models():
        return JSONResponse({"object": "list", "data": model_manager.get_gen_models()})

    @app.get("/health")
    def health():
        pass

    return app
