import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Tutor Chat</title>
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  },
  options: { skipHtmlTags: ['script','noscript','style','textarea'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px;
    background: #f5f5f5;
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  header {
    background: #1a73e8;
    color: white;
    padding: 14px 20px;
    font-size: 17px;
    font-weight: 600;
    letter-spacing: 0.3px;
    flex-shrink: 0;
  }
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .bubble-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
  }
  .bubble-row.student { flex-direction: row-reverse; }
  .avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
    flex-shrink: 0;
  }
  .avatar.tutor  { background: #34a853; color: white; }
  .avatar.student { background: #1a73e8; color: white; }
  .bubble {
    max-width: 72%;
    padding: 10px 14px;
    border-radius: 18px;
    line-height: 1.5;
    word-break: break-word;
  }
  .bubble.tutor {
    background: white;
    border-bottom-left-radius: 4px;
    box-shadow: 0 1px 2px rgba(0,0,0,.12);
    color: #111;
  }
  .bubble.student {
    background: #1a73e8;
    border-bottom-right-radius: 4px;
    color: white;
  }
  .eval-card {
    background: white;
    border-left: 4px solid #ea4335;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,.12);
  }
  .eval-card h3 { color: #ea4335; margin-bottom: 10px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }
  .eval-card table { border-collapse: collapse; width: 100%; }
  .eval-card td { padding: 4px 8px; font-size: 14px; }
  .eval-card td:first-child { color: #555; }
  .eval-card td:last-child { font-weight: 600; color: #111; text-align: right; }
  .typing { display: none; align-items: center; gap: 4px; padding: 4px 0; }
  .typing span {
    width: 7px; height: 7px; border-radius: 50%;
    background: #aaa; animation: bounce 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: .2s; }
  .typing span:nth-child(3) { animation-delay: .4s; }
  @keyframes bounce {
    0%,60%,100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
  }
  #input-area {
    display: flex;
    gap: 8px;
    padding: 12px 16px;
    background: white;
    border-top: 1px solid #e0e0e0;
    flex-shrink: 0;
  }
  #msg {
    flex: 1;
    padding: 10px 14px;
    border: 1px solid #ddd;
    border-radius: 24px;
    font-size: 15px;
    outline: none;
    transition: border-color .2s;
  }
  #msg:focus { border-color: #1a73e8; }
  #send-btn {
    background: #1a73e8;
    color: white;
    border: none;
    border-radius: 24px;
    padding: 10px 20px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: background .2s;
  }
  #send-btn:hover { background: #1557b0; }
  #send-btn:disabled { background: #aaa; cursor: default; }
</style>
</head>
<body>
<header>AI Tutor Chat</header>
<div id="chat">
  <div class="bubble-row" id="typing-row" style="display:none">
    <div class="avatar tutor">T</div>
    <div class="bubble tutor typing" id="typing-indicator" style="display:flex">
      <span></span><span></span><span></span>
    </div>
  </div>
</div>
<div id="input-area">
  <input id="msg" type="text" placeholder="Ask a question..." autocomplete="off">
  <button id="send-btn" onclick="send()">Send</button>
</div>
<script>
let after = 0;
let waitingForTutor = false;

function formatMessage(text) {
  // Escape HTML to prevent injection (browser unescapes in text nodes, so MathJax still sees raw chars)
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  // **bold** → <strong>
  html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  // newlines → <br>
  html = html.replace(/\\n/g, '<br>');
  return html;
}

function showTyping(on) {
  document.getElementById('typing-row').style.display = on ? 'flex' : 'none';
  document.getElementById('send-btn').disabled = on;
  if (on) scrollToBottom();
}

function scrollToBottom() {
  const chat = document.getElementById('chat');
  chat.scrollTop = chat.scrollHeight;
}

function appendMessage(msg) {
  const chat = document.getElementById('chat');
  const typingRow = document.getElementById('typing-row');

  if (msg.role === 'eval') {
    const card = document.createElement('div');
    card.className = 'eval-card';
    let rows = '';
    for (const [k, v] of Object.entries(msg.content)) {
      rows += `<tr><td>${k.replace(/_/g,' ')}</td><td>${v}</td></tr>`;
    }
    card.innerHTML = `<h3>Evaluation</h3><table>${rows}</table>`;
    chat.insertBefore(card, typingRow);
    return card;
  }

  const row = document.createElement('div');
  row.className = `bubble-row ${msg.role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${msg.role}`;
  avatar.textContent = msg.role === 'tutor' ? 'T' : 'S';

  const bubble = document.createElement('div');
  bubble.className = `bubble ${msg.role}`;
  bubble.innerHTML = formatMessage(msg.content);

  row.appendChild(avatar);
  row.appendChild(bubble);
  chat.insertBefore(row, typingRow);
  return bubble;  // return the element that needs MathJax typesetting
}

async function poll() {
  try {
    const res = await fetch(`api/messages?after=${after}`);
    const msgs = await res.json();
    const toTypeset = [];
    for (const msg of msgs) {
      const el = appendMessage(msg);
      if (el) toTypeset.push(el);
      after++;
      if (msg.role === 'tutor' || msg.role === 'eval') {
        waitingForTutor = false;
        showTyping(false);
      }
    }
    if (toTypeset.length > 0) {
      await MathJax.typesetPromise(toTypeset);
      scrollToBottom();
    }
  } catch(e) {
    // server not ready yet or network blip — ignore
  }
  setTimeout(poll, 1000);
}

async function send() {
  const input = document.getElementById('msg');
  const text = input.value.trim();
  if (!text || waitingForTutor) return;
  input.value = '';

  // Show student bubble immediately
  const el = appendMessage({role: 'student', content: text});
  await MathJax.typesetPromise([el]);
  scrollToBottom();

  waitingForTutor = true;
  showTyping(true);

  await fetch('api/send', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text})
  });
}

document.getElementById('msg').addEventListener('keydown', e => {
  if (e.key === 'Enter') send();
});

poll();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, ui, *args, **kwargs):
        self.ui = ui
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = _HTML.encode()
            self._respond(200, "text/html; charset=utf-8", body)
        elif parsed.path in ("/api/messages", "api/messages"):
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
