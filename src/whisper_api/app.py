"""HTTP API. The transcription route follows the shape of OpenAI's /v1/audio/transcriptions.

Audio is held only in a temporary file for the length of the request, and transcripts
are never logged, because both can contain private data.
"""

import asyncio
import logging
import secrets
import threading
import time
from typing import Annotated, Literal

import anyio.to_thread
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, Security, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer
from starlette.datastructures import Headers

from .audio import AudioDecodeError, ChannelMode
from .config import Settings
from .engine import Engine, InvalidOption, TranscribeOptions, TranscriptionCancelled

logger = logging.getLogger("whisper_api")

DISCONNECT_POLL_SECONDS = 1.0


class RequestLoad:
    """Counts requests being uploaded, waiting or transcribed."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.in_flight = 0


class RequestGuard:
    """Checks the key, upload size and queue before any of the upload is accepted.

    This is ASGI middleware rather than a route dependency because FastAPI reads the
    whole body before it runs dependencies.
    """

    def __init__(self, app, api_key: str, max_upload_bytes: int, load: RequestLoad) -> None:
        self.app = app
        self.expected = f"Bearer {api_key}".encode() if api_key else None
        self.max_upload_bytes = max_upload_bytes
        self.load = load

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self.app(scope, receive, send)
            return

        problem = self._check(Headers(scope=scope), scope["method"])
        if problem:
            await problem(scope, receive, send)
            return

        if scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        self.load.in_flight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.load.in_flight -= 1

    def _check(self, headers: Headers, method: str) -> JSONResponse | None:
        if self.expected is not None:
            supplied = headers.get("authorization", "").encode()
            if not secrets.compare_digest(supplied, self.expected):
                return JSONResponse(
                    {"detail": "Missing or wrong API key"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        if method != "POST":
            return None

        # A chunked body overrides Content-Length, which would dodge the size limit.
        length = headers.get("content-length", "")
        if "transfer-encoding" in headers or not length.isdigit():
            return JSONResponse(
                {"detail": "Content-Length is required, chunked uploads are not supported"},
                status_code=411,
            )
        if int(length) > self.max_upload_bytes:
            return JSONResponse({"detail": "Upload is too large"}, status_code=413)
        if self.load.in_flight >= self.load.capacity:
            return JSONResponse(
                {"detail": "Server is busy, try again"},
                status_code=503,
                headers={"Retry-After": "30"},
            )
        return None


def create_app(settings: Settings, engine: Engine) -> FastAPI:
    load = RequestLoad(settings.workers + settings.max_queue)
    app = FastAPI(title="Whisper API", version="0.1.0")
    app.add_middleware(
        RequestGuard,
        api_key=settings.api_key,
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
        load=load,
    )

    slots = asyncio.Semaphore(settings.workers)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": engine.model_name,
            "device": engine.device,
            "compute_type": engine.compute_type,
            "languages": engine.languages,
            "line_gap_seconds": engine.line_gap_seconds,
            "workers": settings.workers,
            "in_flight": load.in_flight,
        }

    # Security() only adds the Authorize button to /docs; RequestGuard does the check.
    @app.post(
        "/v1/audio/transcriptions",
        dependencies=[Security(HTTPBearer(auto_error=False))],
    )
    async def transcribe(
        request: Request,
        file: Annotated[UploadFile, File(description="Audio file")],
        language: Annotated[str | None, Form(description="Force a language, e.g. ms")] = None,
        prompt: Annotated[
            str | None, Form(description="Text hint for the first 30 seconds only")
        ] = None,
        hotwords: Annotated[
            str | None,
            Form(
                description="Terms or a sample sentence, sent with every 30 seconds of audio; "
                "replaces the server default"
            ),
        ] = None,
        channels: Annotated[
            ChannelMode, Form(description="auto splits true stereo; mix makes one signal")
        ] = "auto",
        response_format: Annotated[Literal["json", "text", "verbose_json"], Form()] = "json",
    ) -> Response:
        options = TranscribeOptions(
            language=language or None,
            prompt=prompt,
            hotwords=hotwords,
            channels=channels,
        )
        cancel = threading.Event()
        watcher = asyncio.create_task(_cancel_when_disconnected(request, cancel))
        try:
            async with slots:
                started = time.perf_counter()
                result = await anyio.to_thread.run_sync(
                    engine.transcribe, file.file, options, cancel
                )
        except TranscriptionCancelled:
            logger.info("Request cancelled, the client disconnected")
            return Response(status_code=499)
        except (AudioDecodeError, InvalidOption) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            watcher.cancel()

        took = time.perf_counter() - started
        # Counts and timings only: never log transcript text here.
        logger.info(
            "Transcribed %.1fs of audio in %.1fs (%d segments)",
            result["duration"],
            took,
            len(result["segments"]),
        )

        if response_format == "text":
            return PlainTextResponse(result["text"])
        if response_format == "json":
            return JSONResponse({"text": result["text"]})
        return JSONResponse(
            {
                "task": "transcribe",
                "model": engine.model_name,
                "processing_seconds": round(took, 2),
                **result,
            }
        )

    return app


async def _cancel_when_disconnected(request: Request, cancel: threading.Event) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(DISCONNECT_POLL_SECONDS)
    cancel.set()
