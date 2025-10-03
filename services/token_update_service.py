import logging
import http.server
import socketserver
import json
import threading
from functools import partial

class TokenUpdateHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, orchestrator_service, *args, **kwargs):
        self.orchestrator_service = orchestrator_service
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.path == '/update-tokens':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                tokens = json.loads(post_data)
                
                access_token = tokens.get('access_token')
                refresh_token = tokens.get('refresh_token')

                if not access_token or not refresh_token:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'Missing access_token or refresh_token')
                    return

                self.orchestrator_service.update_tokens(access_token, refresh_token)
                
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Tokens updated successfully')
                logging.info("Tokens successfully updated via HTTP endpoint.")

            except Exception as e:
                logging.error(f"Error handling token update request: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'Internal server error')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging of HTTP requests to keep logs clean
        return

def start_token_update_server(orchestrator_service, start_port=8001, max_retries=10):
    """
    Starts a background HTTP server, finding an open port.
    Returns the server instance and the port number it is running on.
    """
    handler = partial(TokenUpdateHandler, orchestrator_service)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = None
    port = start_port

    for i in range(max_retries):
        port = start_port + i
        try:
            httpd = socketserver.TCPServer(("", port), handler)
            break  # Port is available
        except OSError:
            logging.warning(f"Port {port} is in use, trying next port.")
            httpd = None
            continue
    
    if not httpd:
        logging.error(f"Could not find an open port for the token server after {max_retries} retries.")
        return None, None

    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    logging.info(f"Token update server started in background on port {port}.")
    return httpd, port
