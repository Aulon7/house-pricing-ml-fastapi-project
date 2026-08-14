"""Deployment entrypoint.

The FastAPI CLI discovers an app by looking for main.py, app.py or api.py at
the project root, and `fastapi deploy` gives no way to point at a different
module. The application itself lives in api/main.py; this file only re-exports
it so `fastapi dev`, `fastapi run` and `fastapi deploy` all find it.
"""

from api.main import app

__all__ = ["app"]
