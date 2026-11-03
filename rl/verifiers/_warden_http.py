"""Trusted one-request HTTP worker; never compiles or executes candidate source.

The parent supervises its absolute deadline and reaps this process. This bounds
blocking DNS/TLS/header/body operations without an abandoned network thread.
Request/auth data arrive only over stdin; output is one bounded JSON envelope.
"""

import json
import sys
import time
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_REQUEST_BYTES = 2 * 1024**2
MAX_RESPONSE_BYTES = 2 * 1024**2
MAX_ENVELOPE_BYTES = MAX_RESPONSE_BYTES + 256


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect_disallowed")


def exchange(spec):
    remaining = spec["deadline"] - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    payload = spec["payload"]
    data = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
    if data and len(data) > 1024**2:
        raise ValueError("request_too_large")
    request = Request(spec["url"], data=data, method=spec["method"], headers=spec["headers"])
    # Socket timeout is an additional inactivity limit, not the wall deadline.
    with build_opener(_NoRedirect()).open(request, timeout=remaining) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("response_too_large")
        return {"status": response.status, "body": json.loads(raw)}


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("request_too_large")
        result = exchange(json.loads(raw))
        encoded = json.dumps(
            result, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
        if len(encoded) > MAX_ENVELOPE_BYTES:
            raise ValueError("response_too_large")
    except TimeoutError:
        encoded = b'{"error":"timeout"}'
    except URLError as error:
        encoded = (
            b'{"error":"timeout"}'
            if isinstance(error.reason, TimeoutError)
            else b'{"error":"http_exchange_failed"}'
        )
    except Exception:  # noqa: BLE001 - never expose bearer/request/response data in diagnostics
        encoded = b'{"error":"http_exchange_failed"}'
    sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
