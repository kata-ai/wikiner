# Cross-Lingual Distantly Supervised and Low Resources ID-WikiNER 

## Objectives

Wikipedia as a free online encyclopedia have been proven useful in many Natural Language Processing task and application especially in information extraction and building knowledge base. Wikipedia not only have compacts and rich information for most important and common entities such as people, location and organisation, it also available on multiple languages.
Previous works on multi-lingual Wikipedia with motivation to acquire general corpus and knowledge alignment between high-resources and low-resources language encounter low recall problem. Another work on monolingual data with intensive rules labelling, and label validation also face the same problem [DBPedia and Wikipedia Entity Expansion, and the Modified Rule].
In this research our objective is to bootstrap named entity from Wikipedia in low-resources language as weakly supervised data [1,2,3,4] into training seed or data augmentation. We perform experiment on Wikipedia Indonesia and English languages and test the model performances of automatically tagged NER on different domain.

## Related works

1. [Large-Scale Named Entity Disambiguation Based on Wikipedia Data](http://aclweb.org/anthology/D07-1074) (2007)
2. [Transforming Wikipedia into Named Entity Training Data](http://www.aclweb.org/anthology/U08-1016) (2008)
3. [WiNER: A Wikipedia Annotated Corpus for Named Entity Recognition](http://www.aclweb.org/anthology/I17-1042) — [Github](https://github.com/ghaddarAbs/WiNER) (2017)
4. DBPedia version → http://nerd.eurecom.fr/ontology 
5. [Learning With Annotation Noise](http://www.aclweb.org/anthology/P09-1032) (2009)
6.  [Modified DBpedia Entities Expansion for Tagging Automatically NER Dataset](https://www.researchgate.net/publication/320131070_Modified_DBpedia_Entities_Expansion_for_Tagging_Automatically_NER_Dataset) (2017)
    1. Summary of contribution →  There are 2 modified and 2 new categories for PERSON, while for ORG there is one new category. The paper also proposed 17 rules; consist of 5 modified rules and 12 new rules, which 13 of 17 rules designed for PERSON. In general, M-DEE focused on removing noise (invalid names) in DBpedia for PERSON. 
    2. Missing Link issue →  This need to be done because, after observation to the number of false positive tags in the dataset, we found out that some important places like Indonesia, America and so on did not exist in the Indonesian DBpedia.
    3. Future work mention → Word sense disambiguation to reduce reversed labeling between person and place names that frequently occurs. 
        1. For example as multi-task learning between Cross-lingual Word Sense and NER to reduce ambiguity by related task regularization. (Possible future work todo)
    4. Improving method to detect the candidates of the named entity. [Future work mention]
7.  External data/tools used in the experiment that not in this repository:
    1. [Wikipedia extraction tool](http://medialab.di.unipi.it/wiki/Wikipedia_Extractor) — [github](https://github.com/attardi/wikiextractor)
    2. [Download linked wikipedia](https://github.com/JonathanRaiman/pywikilinks)
    3. [Annotated Wikipedia Extractors](https://github.com/jodaiber/Annotated-WikiExtractor)
    4. [Wikipedia extraction tools 2](https://github.com/JonathanRaiman/wikipedia_ner)
    5. Brat annotation tools

# Cross-Lingual Transfer for Distantly Supervised and Low-resources Indonesian NER

This repository contains the code for our work:

Ikhwantri, Fariz. “Cross-Lingual Transfer for Distantly Supervised and Low-resources Indonesian NER.” ArXiv abs/1907.11158 (2019)

Requirements
============

## Repository Dependency
The model use in this research are based on AllenNLP library 
There are 3 version of AllenNLP version
1. environment-allen.yml Python version 3.7 for experiment using BiLSTM-CRF using AllenNLP library version 0.6
2. environment-dev.yml Python version 3.6.6 for experiment with pycrf-suite (Baseline)
3. environment-exp.yml Python version 3.6.4 for experiment on Multi-Task Sequence Labelling Language Model in [[KS18]](#[KS18])
   1. Kata.ai pytorch re-implementation of https://github.com/marekrei/sequence-labeler 

### preprocessing

preprocess the 20k_dee.txt [[DEE16]](#[DEE16]), 20k_mdee.txt [[MDEE17]](#[MDEE17]), 20k_mdee_gazz.txt [[MDEE17]](#[MDEE17]) and the gold annotation [[GOLD14]](#[GOLD14]) from https://github.com/ialfina/ner-dataset-modified-dee into 2 column conll per sentences (separated by . using several heuristics) and provide the dataset statistics by re-tagging the data into PER, LOC, and ORG tag scheme using scripts/retag_indner.awk


Create a virtual environment from ``environment-{allen, dev, exp}.yml`` file using conda::

    $ conda env create -f environment{allen, dev, exp}.yml

To run experiments with bilm-tf [[MP18]](#[MP18]), Tensorflow is also required.

Dataset
=======

Get our 1200k (1000 train, 200 validation) dataset from [Google Drive LINK]() for seed fine-tuning to reproduce the best results using small amount of data.

Preprocessing for Cross-lingual Transfer
----------------------------------------

<!-- For NeuralSum, the dataset should be further preprocessed using ``prep_oracle_neuralsum.py``::

    $ ./prep_oracle_neuralsum.py -o neuralsum train.01.jsonl

The command will put the oracle files for NeuralSum under ``neuralsum`` directory. Invoke the script with ``-h/--help`` to see its other options. -->

Running experiments
===================

The scripts to run the experiments are named ``run_<model>.py``. For instance, to run an experiment using pycrf-suite, the script to use is ``run_crf.py``. All scripts use `Sacred <https://sacred.readthedocs.io>`_ so you can invoke each with ``help`` command to see its usage. The experiment configurations are fully documented. Run ``./run_<model>.py print_config`` to print all the available configurations and their docs.

Training a model
----------------

To train a model, for example the naive Bayes model, run ``print_config`` command first to see the available configurations::

    

This command will give an output something like::






Evaluating a model
------------------

Evaluating an unsupervised model is simple. For example, to evaluate a LEAD-N summarizer::


This command will print an output like this::

Evaluating a trained model is done similarly with ``model_path`` configuration is set to the path to the saved model.

Setting up Mongodb observer
---------------------------

Sacred allows the experiments to be observed and saved to a Mongodb database. The experiment scripts above can readily be used for this, simply set two environment variables ``SACRED_MONGO_URL`` and ``SACRED_DB_NAME`` to your Mongodb authentication string and database name (to save the experiments into) respectively. Once set, the experiments will be saved to the database. Use ``-u`` flag when invoking the experiment script to disable saving.

Reproducing results
-------------------

All best configurations obtained from tuning on the development set are saved as allenNLP file configuration in config directory

License
=======

Apache License, Version 2.0.

Citation
========

If you're using our code or dataset, please cite::

    @misc{ikhwantri2019crosslingual,
        title={Cross-Lingual Transfer for Distantly Supervised and Low-resources Indonesian NER},
        author={Fariz Ikhwantri},
        year={2019},
        eprint={1907.11158},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

#### [MP18] 
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In Proc. of NAACL. Retrieved from https://www.aclweb.org/anthology/N18-1202/
#### [KS18] 
Kurniawan, Kemal, and Samuel Louvan. "Empirical Evaluation of Character-Based Model on Neural Named-Entity Recognition in Indonesian Conversational Texts." W-NUT 2018 (2018): 85. Retrieved from https://www.aclweb.org/anthology/W18-6112/

### [DEE16]

Ika Alfina, Ruli Manurung, and Mohamad Ivan Fanany, "DBpedia Entities Expansion in Automatically Building Dataset for Indonesian NER", in Proceeding of 8th International Conference on Advanced Computer Science and Information Systems 2016 (ICACSIS 2016).

### [MDEE17]

Ika Alfina, Septiviana Savitri, and Mohamad Ivan Fanany, "Modified DBpedia Entities Expansion for Tagging Automatically NER Dataset", in Proceeding of 9th International Conference on Advanced Computer Science and Information Systems 2017 (ICACSIS 2017).

### [GOLD14]

Andry Luthfi, Bayu Distiawan, and Ruli Manurung, "Building an Indonesian named entity recognizer using Wikipedia and DBPedia", in the Proceesing of 2014 International Conference on Asian Language Processing (IALP)

