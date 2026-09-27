#!/usr/bin/env python3
"""Simple browser-based chat UI for testing prompts without Tkinter.

Usage:
    python simple_chat_ui.py
Then open: http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from prompt_runner import send_prompt

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Lab Prompt Chat</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f3f4f6;
      margin: 0;
      padding: 24px;
    }
    .container {
      max-width: 800px;
      margin: 0 auto;
      background: white;
      border-radius: 12px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.08);
      padding: 16px;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
    }
    select, input, button {
      font-size: 14px;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #d1d5db;
    }
    button {
      background: #2563eb;
      color: white;
      border: none;
      cursor: pointer;
    }
    button:hover { background: #1d4ed8; }
    #chat {
      height: 420px;
      overflow-y: auto;
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 12px;
    }
    .msg {
      margin-bottom: 12px;
      padding: 10px 12px;
      border-radius: 8px;
      white-space: pre-wrap;
    }
    .you { background: #dbeafe; }
    .assistant { background: #dcfce7; }
    .system { background: #fef3c7; }
    .composer {
      display: flex;
      gap: 10px;
    }
    .composer input { flex: 1; }
  </style>
</head>
<body>
  <div class="container">
    <div class="toolbar">
      <label for="mode">Mode:</label>
      <select id="mode">
        <option value="blue">blue</option>
        <option value="red">red</option>
      </select>
    </div>
    <div id="chat"></div>
    <div class="composer">
      <input id="prompt" type="text" placeholder="Type your prompt here..." />
      <button id="send">Send</button>
    </div>
  </div>

  <script>
    const chatEl = document.getElementById('chat');
    const promptEl = document.getElementById('prompt');
    const modeEl = document.getElementById('mode');

    function addMessage(role, text) {
      const box = document.createElement('div');
      box.className = 'msg ' + role;
      box.textContent = role + ': ' + text;
      chatEl.appendChild(box);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    async function sendMessage() {
      const message = promptEl.value.trim();
      if (!message) return;
      promptEl.value = '';
      addMessage('you', message);

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, mode: modeEl.value })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Request failed');
        }
        addMessage('assistant', data.reply || '');
      } catch (error) {
        addMessage('system', String(error.message || error));
      }
    }

    document.getElementById('send').addEventListener('click', sendMessage);
    promptEl.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') sendMessage();
    });

    addMessage('system', 'Connected. Choose blue or red mode and start chatting.');
  </script>
</body>
</html>
"""


class ChatRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != '/api/chat':
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(length) if length > 0 else b''
            payload = json.loads(raw.decode('utf-8') or '{}')
        except Exception as exc:  # pragma: no cover - request parsing
            self.send_response(400)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Invalid JSON: {exc}'}).encode('utf-8'))
            return

        message = str(payload.get('message', '')).strip()
        mode = str(payload.get('mode', 'blue')).strip().lower()
        if not message:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Message is empty.'}).encode('utf-8'))
            return

        try:
            reply = send_prompt(message, mode=mode)
            body = json.dumps({'reply': reply}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(exc)}).encode('utf-8'))

    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output clean.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description='Simple browser chat UI for quick prompt testing.')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the UI to.')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the UI to.')
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ChatRequestHandler)
    print(f"Open http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
