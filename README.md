# WikiNER Distillation

## Objectives

1. Bootstrapping Named Entity scenario from Wikipedia or Clean (Moderated) Unlabeled Data [1,2,3,4]
    1. Wikipedia entities into NER system as training seed augmentation data.
    2. Preliminary work for building Knowledge Base for Question Answering.
2. Reducing or Handling Label noise from self-prediction (soft-label) on automatic data labeling.
3. If we can find decent translation model → It is possible to align cross-lingual entity to use as gazetteer, additional representation e.g: Entity from English language to Indonesia language, consider that Indonesian section sometimes from English ones [9, 16, 17, 18].
    1. incomplete information.
    2. different point of view information. (sentence variation)
    3. missing links on target or source language.
4. Scope Questions:
    1. How many sentences need to be extracted ??
    2. How many need to be corrected by human ??
5. [Optional Future Works] Fine-grained Entity Recognition.

## Repository Dependency

### helfer
setup helfer

```

```


## Prior works

1. [Large-Scale Named Entity Disambiguation Based on Wikipedia Data](http://aclweb.org/anthology/D07-1074) (2007)
2. [Transforming Wikipedia into Named Entity Training Data](http://www.aclweb.org/anthology/U08-1016) (2008)
3. [WiNER: A Wikipedia Annotated Corpus for Named Entity Recognition](http://www.aclweb.org/anthology/I17-1042) — [Github](https://github.com/ghaddarAbs/WiNER) (2017)
    1. [Fine Grained Entity](https://pdfs.semanticscholar.org/7616/3453d7d9f90d23f76bf2a43ea0f786e7d834.pdf?_ga=2.90146843.1088063845.1534736882-1130892961.1508394125)
    2. [Data on google drive](https://drive.google.com/drive/folders/0B6SOo3wyWh6wdGZhYkZDUHRTdkU)
    3. [Wikicoref](http://rali.iro.umontreal.ca/rali/sites/default/files/publis/conll.pdf) — Coreference resolution in Wikipedia, [Rule based alternative](https://github.com/ndass6/Coreference-Resolution)
4. [Exploiting Linguistic Features for Cross-Domain Named Entity Disambiguation](http://www.anlp.jp/proceedings/annual_meeting/2015/pdf_dir/C2-2.pdf) (2015)
5. DBPedia version → http://nerd.eurecom.fr/ontology 
6. [Sequence Knowledge Distillation](https://aclweb.org/anthology/D16-1139) (2016)
7. [Learning With Annotation Noise](http://www.aclweb.org/anthology/P09-1032) (2009)
8. [Training Neural Networks on Noisy Text Annotation](http://www.aclweb.org/anthology/W18-3402) (2018)
    1. Modeling with label flip approach → [Training Deep Neural Networks with Noisy Label](https://arxiv.org/pdf/1406.2080.pdf)
    2. Based on a [noisy adaptation layer](https://openreview.net/forum?id=H12GRgcxg) → Sampling approaches such as EM is not effective, too slow and cannot be integrated into end-to-end training (2017)
    3. Noise Label scalability issue, only generic NER → [Discussion](https://openreview.net/forum?id=H12GRgcxg&noteId=ByZM18kXg)
    4. On a computer vision field → [Learning from Noisy Label with Distillation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Learning_From_Noisy_ICCV_2017_paper.pdf) (2017)
9. [Cross-lingual infobox alignment in Wikipedia using Entity-Attribute Factor Graph](https://iswc2017.semanticweb.org/wp-content/uploads/papers/MainProceedings/76.pdf) (2017) 
10. [Modified DBpedia Entities Expansion for Tagging Automatically NER Dataset](https://www.researchgate.net/publication/320131070_Modified_DBpedia_Entities_Expansion_for_Tagging_Automatically_NER_Dataset) (2017)
    1. Summary of contribution →  There are 2 modified and 2 new categories for PERSON, while for ORG there is one new category. The paper also proposed 17 rules; consist of 5 modified rules and 12 new rules, which 13 of 17 rules designed for PERSON. In general, M-DEE focused on removing noise (invalid names) in DBpedia for PERSON. 
    2. Missing Link issue →  This need to be done because, after observation to the number of false positive tags in the dataset, we found out that some important places like Indonesia, America and so on did not exist in the Indonesian DBpedia.
    3. Future work mention → Word sense disambiguation to reduce reversed labeling between person and place names that frequently occurs. 
        1. For example as multi-task learning between Cross-lingual Word Sense and NER to reduce ambiguity by related task regularization. (Possible future work todo)
    4. Improving method to detect the candidates of the named entity. [Future work mention]
11. [Tri-training for domain adaptation](http://proceedings.mlr.press/v70/saito17a/saito17a.pdf) (2017)
12. [Strong Baselines for Neural Semi-supervised Learning under Domain Shift](https://arxiv.org/pdf/1804.09530.pdf) (2018)
13. External data/tools used in the experiment:
    1. [Wikipedia extraction tool](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor) — [github](https://github.com/attardi/wikiextractor)
    2. [Download linked wikipedia](https://github.com/JonathanRaiman/pywikilinks)
    3. [Wikilinks data](http://wiki-link.nlp2rdf.org/)
    4. [Annotated Wikipedia Extractors](https://github.com/jodaiber/Annotated-WikiExtractor)
    5. [Wikipedia extraction tools 2](https://github.com/JonathanRaiman/wikipedia_ner)
    6. [Polyglot NER](https://arxiv.org/pdf/1410.3791.pdf): [Model](http://polyglot.readthedocs.io/en/latest/NamedEntityRecognition.html)
    7. [Polyglot Embeddings](https://sites.google.com/site/rmyeid/projects/polyglot)
    8. Brat annotation tools
    9. [Bitext alignment tool](https://github.com/clab/fast_align) 
14. Data can be used in experiment:
    1. [YFCC100M Entity Dataset](https://github.com/raingo/yfcc100m-entity) (2017) → Entity Linking Image database
15. [A Study Importance of External Knowledge in Named Entity Recognition Task](http://aclweb.org/anthology/P18-2039) (2018) → Grouping different entity type based on external knowledge (Gazetteer, POS Span Tag)
16. [Cross-lingual Wikification Using Multilingual Embeddings](http://www.aclweb.org/anthology/N16-1072) - ACL Anthology
17. [Cross-lingual Named Entity Recognition via Wikification](http://cogcomp.org/papers/TsaiMaRo16.pdf)
18. [Cheap translation for Cross-lingual Named Entity Recognition](http://cogcomp.org/papers/MayhewTsRo17.pdf)
    1. Use a lexicon to “translate” annotated data available in one or several high resource language(s) into the target language.
    2. Process [Process Bar]
    3. Google Translation improve significantly
19. [A Semi-supervised Algorithm for Indonesian Named Entity Recognition](https://www.computer.org/csdl/proceedings/iscbi/2015/8501/00/8501a045.pdf) (Self-training without Threshold) 2015 
20. Benchmark test on news domain
    1. [Towards Indonesian Part-of-Speech Tagging: Corpus and Model](http://lrec-conf.org/workshops/lrec2018/W34/pdf/3_W34.pdf)
21. [Semi-Supervised Learning Approach for Indonesian  Named Entity Recognition (NER) Using Co-Training Algorithm](https://sci-hub.tw/10.1109/isitia.2016.7828624) 
    1. View 1 dan View 2 are not clear
