import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

class _Handler(BaseHTTPRequestHandler):
    def __init__(self, ui, *args, **kwargs):
        self.ui = ui
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    def _serve_file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                data = f.read()
            self._respond(200, content_type, data)
        except FileNotFoundError:
            self._respond(404, "text/plain", b"Not found")

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_file("static/index.html", "text/html")

        elif parsed.path.startswith("/static/"):
            path = parsed.path.lstrip("/")
            if path.endswith(".css"):
                self._serve_file(path, "text/css")
            elif path.endswith(".js"):
                self._serve_file(path, "application/javascript")

        elif parsed.path.startswith("/api/messages"):
            params = parse_qs(parsed.query)
            after = int(params.get("after", ["0"])[0])
            body = json.dumps(self.ui._get_messages(after)).encode()
            self._respond(200, "application/json", body)
        
        else:
            self._respond(404, "text/plain", b"Not found")

    def do_POST(self):
        if self.path in ("/api/send", "api/send"):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else self.rfile.read()
                data = json.loads(body)
                self.ui.input_queue.put(data["message"])
                self._respond(200, "application/json", b'{"ok":true}')
            except Exception as e:
                print(f"[http] do_POST error: {e}", flush=True)
                self._respond(500, "text/plain", str(e).encode())
        else:
            self._respond(404, "text/plain", b"Not found")

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class WebUI:
    def __init__(self, port=0):
        """
        port=0 lets the OS pick a free port automatically.
        Pass a fixed port (e.g. 8080) if you prefer a predictable URL.
        """
        self.input_queue = queue.Queue()
        self._messages = []
        self._lock = threading.Lock()

        def handler_factory(*args, **kwargs):
            return _Handler(self, *args, **kwargs)

        self._server = ThreadingHTTPServer(("0.0.0.0", port), handler_factory)
        self.port = self._server.server_address[1]

    def add_message(self, role, content):
        """role: 'tutor' | 'student' | 'eval'  (eval content should be a dict)"""
        with self._lock:
            self._messages.append({"role": role, "content": content})

    def _get_messages(self, after=0):
        with self._lock:
            return list(self._messages[after:])

    def start(self):
        """Start the HTTP server in a background daemon thread."""
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        print(f"\n  Tutor chat running at: http://localhost:{self.port}")
        print(f"  Open that URL in your browser (VS Code will offer to forward port {self.port})\n")
