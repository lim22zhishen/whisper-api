"""Start the server: python -m whisper_api"""

import ipaddress
import logging
import sys

import uvicorn

from .app import create_app
from .config import Settings
from .engine import EngineStartupError


def startup_problem(settings: Settings) -> str | None:
    """The server must not be reachable from the network without a key."""
    if settings.api_key or _is_loopback(settings.host):
        return None
    return (
        f"Refusing to listen on {settings.host} without a key. "
        "Set WHISPER_API_KEY in .env, or use WHISPER_HOST=127.0.0.1."
    )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    problem = startup_problem(settings)
    if problem:
        sys.exit(problem)
    try:
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    except EngineStartupError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
