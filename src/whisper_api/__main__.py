"""Start the server: python -m whisper_api"""

import ipaddress
import logging
import sys

import uvicorn

from .app import create_app
from .config import Settings
from .engine import EngineStartupError, load_engine

MIN_NETWORK_KEY_LENGTH = 16


def startup_problem(settings: Settings) -> str | None:
    """The server must not be reachable from the network without a strong key."""
    if len(settings.api_key) >= MIN_NETWORK_KEY_LENGTH or _is_loopback(settings.host):
        return None
    return (
        f"Refusing to listen on {settings.host} without a strong key. Set WHISPER_API_KEY in "
        f".env to at least {MIN_NETWORK_KEY_LENGTH} random characters, "
        "or use WHISPER_HOST=127.0.0.1."
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
        engine = load_engine(settings)
    except EngineStartupError as exc:
        sys.exit(str(exc))
    uvicorn.run(create_app(settings, engine), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
