import subprocess
from typing import List

from SPARQLWrapper import SPARQLWrapper, JSON

idsparql = SPARQLWrapper("http://id.dbpedia.org/sparql")
ensparql = SPARQLWrapper("http://dbpedia.org/sparql")
wikisparql = SPARQLWrapper('https://query.wikidata.org/sparql')

ONTOLOGY_QUERY = 'SELECT ?class ?subclass ?depth { \
	?subclass rdfs:subClassOf ?class. \
	{ \
		SELECT ?subclass (COUNT(?class)-1 AS ?depth) { \
			?subclass rdfs:subClassOf* ?class. \
			?class rdfs:subClassOf* owl:Thing. \
		} \
	} \
} ORDER BY ?depth ?class ?subclass'

SUBCLASS_QUERY = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?person { ?person rdfs:subClassOf* dbo:Person. }'

SUBCLASS_QUERY_WITH_INSTANCES = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?person ?redirectsTo { \
        ?person a/rdfs:subClassOf* dbo:Person. \
        ?person dbo:wikiPageRedirects ?redirectsTo. }'


PERSON_WITH_WIKI = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?name ?redirectsTo { \
        ?ins rdf:type dbpedia-owl:Person. \
        ?ins rdfs:label ?name. \
        ?ins foaf:isPrimaryTopicOf ?redirectsTo. \
    }'

PERSON_WITH_BIRTH = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?name ?redirectsTo ?realName { \
        ?ins rdf:type dbpedia-owl:Person. \
        ?ins rdfs:label ?name. \
        ?ins foaf:isPrimaryTopicOf ?redirectsTo. \
        ?ins dbpedia-owl:birthName ?realName. \
    }'


PERSON_WITH_LINKS = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?name ?redirectsTo ?link { \
        ?ins rdf:type dbpedia-owl:Person. \
        ?ins rdfs:label ?name. \
        ?ins foaf:isPrimaryTopicOf ?redirectsTo. \
        ?ins dbpedia-owl:wikiPageWikiLink ?link. \
        ?link rdf:type dbpedia-owl:Person. \
    }'

PLACE_WITH_WIKI = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?name ?redirectsTo { \
        ?ins rdf:type dbpedia-owl:Place. \
        ?ins rdfs:label ?name. \
        ?ins foaf:isPrimaryTopicOf ?redirectsTo. \
    }'

ORGANISATION_WITH_WIKI = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?name ?redirectsTo { \
        ?ins rdf:type dbpedia-owl:Organisation. \
        ?ins rdfs:label ?name. \
        ?ins foaf:isPrimaryTopicOf ?redirectsTo. \
    }'

PERSON_WITH_WIKI_EN = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?name { \
        ?ins rdf:type dbo:Person. \
        ?ins rdfs:label ?name. } \
        ORDER BY ?name'

PERSON_WITH_WIKI_EN_2 = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?name { \
        ?ins rdf:type dbo:Person. \
        ?ins rdfs:label ?name. \
        FILTER (lang(?name) = "en"). } \
        ORDER BY ?name'

PLACE_WITH_WIKI_EN = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?name { \
        ?ins rdf:type dbo:Place. \
        ?ins rdfs:label ?name. } \
        ORDER BY ?name'

ORGANISATION_WITH_WIKI_EN = 'PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
    PREFIX dbo: <http://dbpedia.org/ontology/> \
    SELECT DISTINCT ?name { \
        ?ins rdf:type dbo:Organisation. \
        ?ins rdfs:label ?name. } \
        ORDER BY ?name'


PERSON_QUERY = "select distinct ?name WHERE { \
        ?ins rdf:type dbpedia-owl:Person. \
        ?ins rdfs:label ?name. }"

ARTIST_QUERY = "PREFIX dbo: <http://dbpedia.org/ontology/> \
    PREFIX dbr: <http://dbpedia.org/resource/> \
    PREFIX dbp: <http://dbpedia.org/property/> \
    SELECT ?singer ?name ?alternativeName \
    WHERE { \
        ?singer rdf:type dbo:MusicalArtist. \
        ?singer dbp:alias ?alternativeName. \
        ?singer rdfs:label ?name. }"

PLACE_QUERY = "select ?name{ \
        ?ins rdf:type dbpedia-owl:Place. \
        ?ins rdfs:label ?name. }"

ORGANISATION_QUERY = "select ?name{ \
        ?ins rdf:type dbpedia-owl:Organisation. \
        ?ins rdfs:label ?name. }"

def findfilename_contains(substring: str, dirpath: str) -> List[str]:
    hosts_process = subprocess.Popen(['egrep','-lir',
                    f'*{substring}*',dirpath.absolute()], stdout= subprocess.PIPE)
    filenames, _ = hosts_process.communicate()
    filenames = filenames.decode('utf-8').split('\n')
    return filenames


def query_all(sparql: SPARQLWrapper, query: str, limit: int = 10000, verbose=False):
    numres, offset, all_results = 10000, 0, []
    while numres >= limit:
        
        sparql.setQuery(query + f' OFFSET {offset}')
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if verbose:
            for res in results["results"]["bindings"]:
                print(res)
        numres = len(results["results"]["bindings"])
        print(f'result count: {numres}, offset: {offset}')
        all_results.extend(results["results"]["bindings"])
        offset = offset + numres
    print(f'{query} count {len(all_results)}')
    return all_results

def query_all_stream(sparql: SPARQLWrapper, query: str, limit: int = 10000, verbose=False):
    numres, offset = 10000, 0
    while numres >= limit:
        sparql.setQuery(query + f' OFFSET {offset}')
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        numres = len(results["results"]["bindings"])
        offset = offset + numres
        for res in results["results"]["bindings"]:
            yield res

def generic_query(sparql: SPARQLWrapper, query: str):
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results


if __name__ == '__main__':
    # Test query result
    query_all(ensparql, PERSON_WITH_WIKI_EN, verbose=True)
    # query_all(idsparql, PLACE_WITH_WIKI, verbose=True)
    # query_all(ensparql, ORGANISATION_WITH_WIKI_EN, verbose=True)
