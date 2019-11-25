import re
import os
import sys
import argparse
import codecs
import glob
import json

from wikipediaapi import Wikipedia
from wikipediaapi import WikipediaPage
from wikipediaapi import ExtractFormat

def word_tokenize(sentence, sep=r'(\W+)?'):
    return [x.strip() for x in re.split(sep, sentence) if x.strip()]


def get_jsonlpage(data):
    for line in data.readlines():
        json_line = json.loads(line)
        if 'text' in json_line:
            title = json_line['text'].split(".\n")[0]
            yield title

def print_page(page: WikipediaPage, extract_format: ExtractFormat, 
               section_sep="\n\n", sentence_sep="\n"):
    no_print_sec = ['references', 'see also', 'links', 'list']
    txt = page.summary
    if txt:
        txt += section_sep

    def combine(sections, level):
        res = ""
        for sec in sections:
            if sec.title.lower() in no_print_sec or any([nps in sec.title.lower() for nps in no_print_sec]):
                continue
            # if extract_format == ExtractFormat.WIKI:
            #     res += sec.title
            # else:
                # raise NotImplementedError("Unknown ExtractFormat type")
            res += "\n"
            content = re.sub('<li|.*?/li>', '', sec.text)
            res += content
            if sec.text:
                res += section_sep

            res += combine(sec.sections, level + 1)

        return res

    txt += combine(page.sections, 2)

    return txt

def main(args):
    files = glob.glob(args.filepattern)
    id_wiki = Wikipedia(language='id')
    en_wiki = Wikipedia(language='en', extract_format=args.format)
    for corpus in files:
        print(corpus)
        if os.path.isfile(corpus):
            _, fname = os.path.split(corpus)
            if args.output_dir and os.path.isdir(args.output_dir):
                output_file = os.path.join(args.output_dir, fname)
                mode = 'w+'
                print(output_file)
                if os.path.exists(output_file) and args.duplicate_append:
                    print('file exists')
                    mode = 'a'
                fileout = codecs.open(output_file, mode=mode, encoding=args.encoding)
            else:
                fileout = sys.stdout
            data = codecs.open(corpus, mode='r', encoding=args.encoding)
            for title in get_jsonlpage(data):
                page = id_wiki.page(title)
                print(title)
                try:
                    # print(page.langlinks)
                    if 'en' in page.langlinks:
                        en_title = page.langlinks['en'].title
                        en_page = en_wiki.page(en_title)
                        print(en_title)
                        # print(en_page.text)
                        en_text = print_page(en_page, args.format)
                        print(en_text, file=fileout)
                except Exception:
                    continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""get interlanguage aligned content/title from a wikipedia dump using wikipedia API""")
    parser.add_argument('filepattern', metavar='FILE', help='path to the corpus file')
    parser.add_argument('-o', '--output-dir', metavar='DIR', default=None,
                        help='output directory (default: {})'.format(None))
    parser.add_argument('--format',
                        default=ExtractFormat.WIKI,
                        type=int,
                        const=ExtractFormat.WIKI,
                        nargs='?',
                        choices=[ExtractFormat.WIKI, ExtractFormat.HTML],
                        help='list output format (default: %(default)s)')
    parser.add_argument('--encoding', default='utf-8', help='file encoding (default: utf-8)')
    args = parser.parse_args()
    main(args)
