import http.server
import socketserver
import os

PORT = 8000

print(f"localhost:{PORT}")

Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map.update(
    {
        ".js": "application/x-javascript",
    }
)

os.chdir("static")
httpd = socketserver.TCPServer(("", PORT), Handler)
httpd.serve_forever()
