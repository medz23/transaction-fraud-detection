
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded


def bad_input(request: Request, exc: RequestValidationError):
    details = [f"{'.'.join(map(str, e['loc'][1:]))}: {e['msg']}" for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": details})


def too_many(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "rate_limited", "detail": str(exc.detail)})


def crashed(request: Request, exc: Exception):
    logging.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "something went wrong"})


def add_handlers(app: FastAPI):
    app.add_exception_handler(RequestValidationError, bad_input)
    app.add_exception_handler(RateLimitExceeded, too_many)
    app.add_exception_handler(Exception, crashed)
