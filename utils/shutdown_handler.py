import threading
import http.server
import socketserver
import logging
import socket

# Global event for graceful shutdown
SHUTDOWN_EVENT = threading.Event()
httpd_server = None # Global reference to the HTTP server

class ShutdownHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/shutdown':
            logging.info("Received shutdown request via HTTP.")
            SHUTDOWN_EVENT.set()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"DGN Client shutting down.")
            if httpd_server:
                logging.info("ShutdownHandler: Attempting to shut down HTTP server.")
                threading.Thread(target=httpd_server.shutdown, daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()

def find_free_port():
    """Finds and returns an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_shutdown_server():
    """Starts a simple HTTP server in a new thread to listen for shutdown requests."""
    handler = ShutdownHandler
    global httpd_server
    
    port = find_free_port()
    # We need to let the parent process know which port we are using
    print(f"DGN_CLIENT_SHUTDOWN_SERVER_PORT:{port}", flush=True)

    with socketserver.TCPServer(('', port), handler, bind_and_activate=False) as httpd:
        httpd.allow_reuse_address = True
        httpd_server = httpd
        try:
            httpd.server_bind()
            httpd.server_activate()
            logging.info(f"Shutdown server started on port {port}")
            httpd.serve_forever()
        except Exception as e:
            if not SHUTDOWN_EVENT.is_set():
                logging.error(f"Failed to start shutdown server: {e}")
        finally:
            httpd.server_close()
            logging.info("Shutdown server stopped.")
            httpd_server = None
