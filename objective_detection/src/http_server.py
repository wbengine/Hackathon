import cgi
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import io
import json
import cv2
import numpy as np
import time

from model import Model

model = Model().prepare()


# class CORSRequestHandler (SimpleHTTPRequestHandler):
#     def end_headers (self):
#         self.send_header('Access-Control-Allow-Origin', '*')
#         SimpleHTTPRequestHandler.end_headers(self)


class PostHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        # Parse the form data posted
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': self.headers['Content-Type'],
            }
        )

        # out.write('Client: {}\n'.format(self.client_address))
        # out.write('User-agent: {}\n'.format(
        #     self.headers['user-agent']))
        # out.write('Path: {}\n'.format(self.path))
        # out.write('Form data:\n')

        # Echo back information about what was posted in the form
        field_item = form['image']
        if field_item.filename:
            # The field contains an uploaded file
            file_data = field_item.file.read()

            print('accept image %s' % field_item.filename)
            time_beginning = time.time()
            array = np.frombuffer(file_data, dtype='uint8')
            frame = cv2.imdecode(array, 1)
            res_list, _ = model.run(frame)
            print('finished recognition (time={:.2}s) {}'.format(
                (time.time() - time_beginning),
                json.dumps(res_list, indent=2)))

            # Begin the response
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type',
                             'text/plain; charset=utf-8')
            self.end_headers()

            out = io.TextIOWrapper(
                self.wfile,
                encoding='utf-8',
                line_buffering=False,
                write_through=True,
            )
            out.write(json.dumps(res_list))
            # Disconnect our encoding wrapper from the underlying
            # buffer so that deleting the wrapper doesn't close
            # the socket, which is still being used by the server.
            out.detach()
        else:
            print('[Error] accept an unknown post !')
            self.send_error(400)


if __name__ == '__main__':
    from http.server import HTTPServer

    server = HTTPServer(('0.0.0.0', 8080), PostHandler)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()
