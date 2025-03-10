from typing import Sequence, Callable, Optional

from aiohttp.web import (
    Application as AiohttpApplication,
    View as AiohttpView,
    Request as AiohttpRequest,
)
from aiohttp_apispec import setup_aiohttp_apispec
from aiohttp_session.cookie_storage import EncryptedCookieStorage
from aiohttp_session import setup as session_setup
import aiohttp_cors
from app.user.dataclasses import UserDC

from .config import Config, setup_config
from .logger import setup_logging
from .middlewares import setup_middlewares
from .routes import register_urls


__all__ = ("Application",)

from app.store import Store, setup_store
from app.store.database.database import Database


class Application(AiohttpApplication):
    config: Optional[Config] = None
    store: Optional[Store] = None
    database: Optional[Database] = None


class Request(AiohttpRequest):
    user: UserDC | None = None

    @property
    def app(self) -> Application:
        return super().app()


class View(AiohttpView):
    @property
    def request(self) -> Request:
        return super().request

    @property
    def database(self):
        return self.request.app.database

    @property
    def store(self) -> Store:
        return self.request.app.store

    @property
    def data(self) -> dict:
        return self.request.get("data", {})


app = Application()


def setup_app(config_path: str) -> Application:
    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        },
    )

    setup_logging(app)
    setup_config(app, config_path)
    session_setup(app, EncryptedCookieStorage(app.config.session.key))
    register_urls(application=app, cors=cors)
    setup_aiohttp_apispec(
        app, title="Svoyak bot", url="/docs/json", swagger_path="/docs"
    )
    setup_middlewares(app)
    setup_store(app)
    return app
