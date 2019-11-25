"""
Filter Automatically Tagged sentence from valid wikipedia entry only
"""
import sys
import json
from urllib.error import HTTPError

from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

from entity_query import generic_query, idsparql, ensparql


IS_PERSON = 'ASK { dbpedia-id:[QUERY] rdf:type foaf:Person }'
IS_PLACE = 'ASK { dbpedia-id:[QUERY] rdf:type dbpedia-owl:Place }'
IS_ORGANISATION = 'ASK { dbpedia-id:[QUERY] rdf:type dbpedia-owl:Organisation }'

IS_ENTITY = 'ASK { {dbpedia-id:[QUERY] rdf:type foaf:Person.} \
        UNION {dbpedia-id:[QUERY] rdf:type dbpedia-owl:Place.} \
        UNION {dbpedia-id:[QUERY] rdf:type dbpedia-owl:Organisation} \
    }'

IS_ENTITY_EN = 'ASK { {<http://dbpedia.org/resource/[QUERY]> rdf:type foaf:Person.} \
        UNION {<http://dbpedia.org/resource/[QUERY]> rdf:type <http://dbpedia.org/ontology/Person>.} \
        UNION {<http://dbpedia.org/resource/[QUERY]> rdf:type <http://dbpedia.org/ontology/Place>.} \
        UNION {<http://dbpedia.org/resource/[QUERY]> rdf:type <http://dbpedia.org/ontology/Organisation>.} \
    }'

ID_ENTITY_EN = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
        SELECT ?en_name WHERE { \
        <http://id.dbpedia.org/resource/[QUERY]> owl:sameAs ?en_name\
    }'

ID_TO_EN = 'select ?en_name { \
        values dbpedia-owl:Thing <http://id.dbpedia.org/resource/[QUERY]> \
        <http://id.dbpedia.org/resource/[QUERY]> owl:sameAs ?en_name. filter langMatches(lang(?en_name),en) \
    }'


GET_TYPE = 'SELECT ?type WHERE { \
        dbpedia-id:[QUERY] rdf:type ?type. \
    }'

GET_TYPE_EN = ' PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
        SELECT ?type \
        WHERE { <http://dbpedia.org/resource/[QUERY]> rdf:type ?type }'


GET_ALL_PROPS = 'SELECT * WHERE { \
        dbpedia-id:[QUERY] ?p ?o. \
    }'

WIKIDATA_QUERY = 'SELECT DISTINCT ?c WHERE { \
  ?Q wdt:P31/wdt:P279? ?c . \
  ?Q rdfs:label "[QUERY]"@id \
}'

WIKIDATA_FILTER = 'ASK { \
  ?Q rdfs:label "[QUERY]"@id.\
  ?Q wdt:P31/wdt:P279? wd:Q5/Q215627}'

IS_A_ENTITY = 'ASK { \
  ?Q rdfs:label "[QUERY]"@id. \
  ?t rdfs:label "[TYPE]"@en. \
  ?Q wdt:P31/wdt:P279? ?t}'

GET_EN_LABEL = 'SELECT DISTINCT ?label WHERE { \
  ?Q rdfs:label "[QUERY]"@id . \
  ?Q rdfs:label ?label. \
  FILTER (lang(?label) = "en") } '

GET_EN_LABEL_DBPEDIA = 'SELECT DISTINCT ?label WHERE { \
  ?Q owl:sameAs dbpedia-id:[QUERY] . \
  ?Q rdfs:label ?label. \
  FILTER (lang(?label) = "en").}'


def get_annotation_type(tag_uri, sparql, get_query: str = GET_TYPE, link_type: bool = False):
    annotation_type = set()
    tag_result = generic_query(sparql, get_query.replace('[QUERY]', tag_uri))
    for rdf_type in tag_result['results']['bindings']:
        if link_type:
            ret_type = rdf_type['type']['value']
        else:
            # get last path of rdf type as possible entity
            ret_type = rdf_type['type']['value'].rsplit('/', 1)[-1]
            ret_type = ret_type.split('_', 1)[-1]
        annotation_type.add(ret_type)
    return annotation_type

def is_entity(tag_uri, sparql, is_query: str = IS_ENTITY):
    """
    tag_uri: str
        check if tag_uri is a membership of an entity return 
        True if the tag is an entity
        False if the tag is not an entity or the uri is a bad query
    """
    try:
        # validate extracted uri is valid entry on kb such using sparql
        results = generic_query(sparql, is_query.replace('[QUERY]', tag_uri))
        return results['boolean']
    except QueryBadFormed:
        return False
    except HTTPError:
        return False
    return False


def read_annotated_wiki(filename: str, encoding='utf-8', sparql=None, link_type=False):
    """
    filename: str
        jsonline file contain url, text, id and annotations from Annotated Wikipedia Extractor
        example
            {"url": "http://en.wikipedia.org/wiki/Anarchism", 
             "text": "Anarchism.\nAnarchism is a political philosophy which considers the state 
                undesirable, unnecessary and harmful, and instead promotes a stateless society, or 
                anarchy. It seeks to diminish ...", 
             "id": 12, 
             "annotations": [
                {"to": 46, "from": 26, "id": "Political_philosophy", "label": "political philosophy"}, 
                {"to": 72, "from": 67, "id": "State_(polity)", "label": "state"},
                ...
                {"to": 163, "from": 156, "id": "Anarchy", "label": "anarchy"}, 
            ]}
    encoding
        file encoding to read files
    """
    with open(filename, mode='r', encoding=encoding) as f:
        for line in f:
            doc = json.loads(line.strip())
            title = doc['text'].split('\n')[0]
            title = ''.join(title.split('.')[:-1])
            print(title)
            kb_tag = []
            for tag in doc['annotations']:
                # validate extracted uri entry on a kb such as dbpedia using sparql
                try:
                    tag_uri = tag['uri']
                    if is_entity(tag_uri, sparql):
                        tag['type'] = get_annotation_type(tag_uri, sparql, link_type=link_type)
                        kb_tag.append(tag)
                    # print(GET_EN_LABEL_DBPEDIA.replace('[QUERY]',tag_uri))
                    res = generic_query(ensparql, 
                                        GET_EN_LABEL_DBPEDIA.replace('[QUERY]',tag_uri))
                    # print(res)
                    if res['results']['bindings']:
                        # print(res['results']['bindings'])
                        en_tag_uri = res['results']['bindings'][0]['label']['value']
                        # print(en_tag_uri)
                        if is_entity(en_tag_uri, ensparql, is_query=IS_ENTITY_EN):
                            tag['en_type'] = get_annotation_type(en_tag_uri, ensparql, \
                                                                 get_query=GET_TYPE_EN, \
                                                                 link_type=link_type)
                        kb_tag.append(tag)
                except QueryBadFormed:
                    pass
                except HTTPError:
                    pass
            if kb_tag:
                new_doc = { 'title': title,
                    'text': doc['text'],
                    'annotations': kb_tag,
                    'url': doc['url'],
                }
                title_uri = title.replace(' ', '_')
                try:
                    if is_entity(title_uri, sparql):
                        new_doc['doc_type'] = get_annotation_type(title_uri, sparql, link_type=link_type)
                        new_doc['title_uri'] = title_uri
                        print(new_doc['doc_type'])
                    if is_entity(title_uri, ensparql, is_query=IS_ENTITY_EN):
                        new_doc['en_doc_type'] = get_annotation_type(title_uri, ensparql, \
                                                                     get_query=GET_TYPE_EN, \
                                                                     link_type=link_type)
                        new_doc['title_uri'] = title_uri
                        doc_type = new_doc['en_doc_type']
                        print(f'EN: {doc_type}')  
                except QueryBadFormed:
                    pass
                except HTTPError:
                    pass
                yield new_doc


if __name__ == '__main__':
    entity_wiki = read_annotated_wiki(sys.argv[1], sparql=idsparql)
    for wiki_doc in entity_wiki:
        # print(wiki_doc)
        pass
