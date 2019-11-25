
'''
Simple tagger service using sklearn_crfsuite.

Author:     Fariz Ikhwantri
Version:    2017-12-01
'''

from argparse import ArgumentParser
from cgi import FieldStorage

# import spacy

from json import dumps

from sys import version as python_version

from http.server import BaseHTTPRequestHandler, HTTPServer

import sklearn_crfsuite

from ingredients.commons import tag
from ingredients.commons import word_tokenize
from ingredients.crf_utils import sent2features

# Constants
TAGGER = None
SEP = None



class CrfsuiteTaggerHandler(BaseHTTPRequestHandler):
    global TAGGER
    def do_POST(self):
        if python_version.startswith('3'):
            self.do_POSTv3()
        else:
            self.do_POSTv2()

    def do_POSTv2(self):
        print('Received request')
        field_storage = FieldStorage(
                headers=self.headers,
                environ={
                    'REQUEST_METHOD':'POST',
                    'CONTENT_TYPE':self.headers['Content-Type'],
                    },
                fp=self.rfile)
 
        json_dict = tag_to_json(TAGGER,field_storage.value.decode('utf-8','ignore'))

        # Write the response
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()

        self.wfile.write(dumps(json_dict))
        print(('Generated %d annotations' % len(json_dict)))
    
    def do_POSTv3(self):
        print('Received request')
        request_headers = self.headers
        content_length = request_headers.get("Content-length")
        length = int(content_length) if content_length else 0
        message = self.rfile.read(length)
        print(message)
        # postvars = self.parse_POST()
        json_dict = tag_to_json(TAGGER, message)

        # Write the response
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()

        self.wfile.write(dumps(json_dict).encode())
        # dump(json_dict, self.wfile)
        print('Generated %d annotations' % len(json_dict))

    def log_message(self, format, *args):
        return # Too much noise from the default implementation


def tag_to_json(tagger, text, sep="\n", window_size=0):
    annotations = {}

    def _add_ann(start, end, _type):
        annotations[len(annotations)] = {
            'type': _type,
            'offsets': ((start, end), ),
            'texts': ((text[start:end]), ),
        }
    print(text)
    text = text.decode("utf-8")
    print(SEP)
    data = text.split(SEP)
    # data = re.split(SEP, text)
    # data = text.split(sep)
    data = [x for x in data if x]
    length = 0
    for sent in data:
        print("sent : ", sent)
        x_feat = sent2features(word_tokenize(sent), window_size)
        result = tagger.predict(x_feat)
        result = tag(sent, sent.split(), result)
        print(result)
        for span in result:
            if span["tagname"] != "O":
                start = length + int(span["start"])
                end = length + int(span["end"])
                # print(start)
                # print(end)
                # print(text[start:end])
                _add_ann(start, end, span["tagname"])
        length += len(sent+sep)

    return annotations


def main(args):
    global TAGGER
    global SEP
    
    # TAGGER = CRFTagger(args.model_file)

    print(f'load from: {args.model_file}.pkl')
    TAGGER = sklearn_crfsuite.CRF(model_filename=args.model_file)
    print('Done!')
    
    SEP = args.separator
    print(SEP)

    server_class = HTTPServer
    httpd = server_class(('localhost', args.port), CrfsuiteTaggerHandler)
    print('crf tagger service started')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print('crf tagger service stopped')


if __name__ == '__main__':
    # from sys import argv
    parser = ArgumentParser(description='crf suite brat tagger')
    parser.add_argument('-p', '--port', type=int, default=47111,
                        help='port to run the HTTP service on (default: 47111)')
    parser.add_argument('-m', '--model-file', type=str, default='crf',
                        help='binary file of trained crf suite model')
    parser.add_argument('-d', '--data-train', type=str,
                        help='train json file consist of text and label as key')
    parser.add_argument('-s', '--separator', type=str, default="\n",
                        help='sentence separator')
    argp = parser.parse_args()
    exit(main(argp))
