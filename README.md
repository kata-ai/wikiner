# WikiNER Distillation

## Objectives

Wikipedia as a free online encyclopedia have been proven useful in many Natural Language Processing task and application especially in information extraction and building knowledge base. Wikipedia not only have compacts and rich information for most important and common entities such as people, location and organisation, it also available on multiple languages.
Previous works on multi-lingual Wikipedia with motivation to acquire general corpus and knowledge alignment between high-resources and low-resources language encounter low recall problem. Another work on monolingual data with intensive rules labelling, and label validation also face the same problem [DBPedia and Wikipedia Entity Expansion, and the Modified Rule].
In this research our objective is to bootstrap named entity from Wikipedia in low-resources language as weakly supervised data [1,2,3,4] into training seed or data augmentation. We perform experiment on Wikipedia Indonesia and English languages and test the model performances of automatically tagged NER on different domain.

## Repository Dependency
The model use in this research are based on AllenNLP library 
There are 3 version of AllenNLP version
1. environment-allen.yml Python version 3.7 for experiment using AllenNLP version 0.6
2. environment-dev.yml Python version 3.6.6 for experiment with pycrf-suite (Baseline)
3. environment-exp.yml Python version 3.6.4 for experiment on Multi-Task Sequence Labelling Language Model 
   1. seqlab (Internal Researcb Code)

### helfer


## Prior works

1. [Large-Scale Named Entity Disambiguation Based on Wikipedia Data](http://aclweb.org/anthology/D07-1074) (2007)
2. [Transforming Wikipedia into Named Entity Training Data](http://www.aclweb.org/anthology/U08-1016) (2008)
3. [WiNER: A Wikipedia Annotated Corpus for Named Entity Recognition](http://www.aclweb.org/anthology/I17-1042) — [Github](https://github.com/ghaddarAbs/WiNER) (2017)
4. DBPedia version → http://nerd.eurecom.fr/ontology 
5. [Sequence Knowledge Distillation](https://aclweb.org/anthology/D16-1139) (2016)
6. [Learning With Annotation Noise](http://www.aclweb.org/anthology/P09-1032) (2009)
7.  [Modified DBpedia Entities Expansion for Tagging Automatically NER Dataset](https://www.researchgate.net/publication/320131070_Modified_DBpedia_Entities_Expansion_for_Tagging_Automatically_NER_Dataset) (2017)
    1. Summary of contribution →  There are 2 modified and 2 new categories for PERSON, while for ORG there is one new category. The paper also proposed 17 rules; consist of 5 modified rules and 12 new rules, which 13 of 17 rules designed for PERSON. In general, M-DEE focused on removing noise (invalid names) in DBpedia for PERSON. 
    2. Missing Link issue →  This need to be done because, after observation to the number of false positive tags in the dataset, we found out that some important places like Indonesia, America and so on did not exist in the Indonesian DBpedia.
    3. Future work mention → Word sense disambiguation to reduce reversed labeling between person and place names that frequently occurs. 
        1. For example as multi-task learning between Cross-lingual Word Sense and NER to reduce ambiguity by related task regularization. (Possible future work todo)
    4. Improving method to detect the candidates of the named entity. [Future work mention]
8.  External data/tools used in the experiment:
    1. [Wikipedia extraction tool](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor) — [github](https://github.com/attardi/wikiextractor)
    2. [Download linked wikipedia](https://github.com/JonathanRaiman/pywikilinks)
    3. [Wikilinks data](http://wiki-link.nlp2rdf.org/)
    4. [Annotated Wikipedia Extractors](https://github.com/jodaiber/Annotated-WikiExtractor)
    5. [Wikipedia extraction tools 2](https://github.com/JonathanRaiman/wikipedia_ner)
    6. [Polyglot NER](https://arxiv.org/pdf/1410.3791.pdf): [Model](http://polyglot.readthedocs.io/en/latest/NamedEntityRecognition.html)
    7. [Polyglot Embeddings](https://sites.google.com/site/rmyeid/projects/polyglot)
    8. Brat annotation tools
    9. [Bitext alignment tool](https://github.com/clab/fast_align) 
9.  [A Study Importance of External Knowledge in Named Entity Recognition Task](http://aclweb.org/anthology/P18-2039) (2018) → Grouping different entity type based on external knowledge (Gazetteer, POS Span Tag)
10. [Cross-lingual Wikification Using Multilingual Embeddings](http://www.aclweb.org/anthology/N16-1072) - ACL Anthology
11. [Cross-lingual Named Entity Recognition via Wikification](http://cogcomp.org/papers/TsaiMaRo16.pdf)
12. [Cheap translation for Cross-lingual Named Entity Recognition](http://cogcomp.org/papers/MayhewTsRo17.pdf)
    1. Use a lexicon to “translate” annotated data available in one or several high resource language(s) into the target language.
    2. Process [Process Bar]
    3. Google Translation improve significantly
13. [A Semi-supervised Algorithm for Indonesian Named Entity Recognition](https://www.computer.org/csdl/proceedings/iscbi/2015/8501/00/8501a045.pdf) (Self-training without Threshold) 2015 
14. Benchmark test on news domain
    1. [Towards Indonesian Part-of-Speech Tagging: Corpus and Model](http://lrec-conf.org/workshops/lrec2018/W34/pdf/3_W34.pdf)
15. [Semi-Supervised Learning Approach for Indonesian  Named Entity Recognition (NER) Using Co-Training Algorithm](https://sci-hub.tw/10.1109/isitia.2016.7828624) 
    1. View 1 dan View 2 are not clear
