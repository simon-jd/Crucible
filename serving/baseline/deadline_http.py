"""Linux main-thread deadlines for the two pinned acquisition paths only."""

import contextlib
import http.client
import ipaddress
import json
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("acquisition absolute deadline exceeded")
    return value


@contextlib.contextmanager
def absolute_deadline(deadline):
    """Interrupt blocking Python socket/file phases; never steal another alarm."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("acquisition deadline requires the main thread")
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    if (
        signal.SIGALRM in old_mask
        or signal.SIGALRM in signal.sigpending()
        or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0)
        # A pre-existing timer can expire between the first pending check and
        # getitimer. Recheck while still blocked before claiming ownership.
        or signal.SIGALRM in signal.sigpending()
    ):
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        raise RuntimeError("acquisition refuses an active/blocked/pending interval timer")
    old_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("acquisition absolute deadline exceeded")

    try:
        delay = remaining(deadline)
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, delay)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        yield
        remaining(deadline)
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Consume our pending alarm while blocked, before restoring the caller's
        # handler. A deadline firing during teardown must not reach that handler.
        while signal.sigtimedwait({signal.SIGALRM}, 0) is not None:
            pass
        signal.signal(signal.SIGALRM, old_handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        # The prior timer was inactive; it stays inactive. No worker/timer thread.


_RESOLVE = """
import json,socket,sys
rows=socket.getaddrinfo(sys.argv[1],int(sys.argv[2]),0,socket.SOCK_STREAM)
if len(rows)>64: raise RuntimeError('DNS address bound exceeded')
print(json.dumps(rows))
"""


def resolve(host, port, deadline):
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # getaddrinfo may block in libc beyond a Python signal. An exact owned
        # subprocess is killable/reaped by subprocess.run on timeout/BaseException.
        try:
            result = subprocess.run(
                [sys.executable, "-B", "-S", "-c", _RESOLVE, host, str(port)],
                capture_output=True,
                timeout=remaining(deadline),
                check=True,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError("DNS absolute deadline exceeded") from None
        remaining(deadline)
        if len(result.stdout) > 16384:
            raise ValueError("DNS response bound exceeded")
        rows = json.loads(result.stdout)
        if not rows or len(rows) > 64:
            raise ValueError("DNS address bound exceeded")
        return rows
    family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
    target = (host, port, 0, 0) if address.version == 6 else (host, port)
    return [(family, socket.SOCK_STREAM, 0, "", target)]


def connect(address, deadline, source_address=None):
    last = None
    for family, kind, protocol, _, target in resolve(*address, deadline):
        sock = socket.socket(family, kind, protocol)
        try:
            sock.settimeout(min(30, remaining(deadline)))
            if source_address:
                sock.bind(source_address)
            sock.connect(tuple(target))
            remaining(deadline)
            return sock
        except BaseException as error:
            sock.close()
            if not isinstance(error, OSError) or isinstance(error, TimeoutError):
                raise
            last = error
    raise last or OSError("no usable DNS address")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def open_response(url, deadline, allowed):
    """Verified TLS/default trust, no proxies, no redirect/error body reads."""

    def factory(cls):
        def create(host, **kwargs):
            connection = cls(host, **kwargs)
            connection._create_connection = lambda address, timeout=None, source_address=None: (
                connect(address, deadline, source_address)
            )
            return connection

        return create

    class HTTP(urllib.request.HTTPHandler):
        def http_open(self, request):
            return self.do_open(factory(http.client.HTTPConnection), request)

    class HTTPS(urllib.request.HTTPSHandler):
        def https_open(self, request):
            return self.do_open(
                factory(http.client.HTTPSConnection), request, context=self._context
            )

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), HTTP(), HTTPS(), _NoRedirect()
    )
    for hop in range(6):
        parsed = urllib.parse.urlparse(url)
        if parsed.username or parsed.password or not allowed(parsed):
            raise ValueError("unreviewed acquisition redirect/host")
        request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
        try:
            response = opener.open(request, timeout=min(30, remaining(deadline)))
            try:
                remaining(deadline)
            except BaseException:
                response.close()
                raise
            return response
        except urllib.error.HTTPError as response:
            try:
                if response.code not in (301, 302, 303, 307, 308):
                    raise ValueError(f"acquisition HTTP status {response.code}") from None
                if hop == 5 or not response.headers.get("Location"):
                    raise ValueError("acquisition redirect bound/location")
                url = urllib.parse.urljoin(url, response.headers["Location"])
            finally:
                response.close()
        except urllib.error.URLError:
            remaining(deadline)  # Convert an expired connect/TLS wrapper to timeout.
            raise
    raise AssertionError("unreachable redirect state")
