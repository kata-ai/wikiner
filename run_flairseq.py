from flair.embeddings import CharLMEmbeddings, StackedEmbeddings
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.trainers import SequenceTaggerTrainer
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.data import TaggedCorpus
import os

### read data ###
# use your own data path
dirs = os.path.dirname(os.path.abspath(__file__))
train_corpus = 'data-train.conll'
dev_corpus = 'data-dev.conll'
test_corpus = 'data-test.conll'
data_path = dirs + '/data-all/'
model_path = dirs + '/models/'
which_lm = "kompas+tempo"

# get training, test and dev data
sentences_train = NLPTaskDataFetcher.read_conll_2_column_data(data_path + train_corpus, "NER")
sentences_dev = NLPTaskDataFetcher.read_conll_2_column_data(data_path + dev_corpus, "NER")
sentences_test = NLPTaskDataFetcher.read_conll_2_column_data(data_path + test_corpus, "NER")

# for token in sentences_dev[0].tokens:
#   print(token.get_tag("NER"))
# return corpus
corpus: TaggedCorpus = TaggedCorpus(sentences_train, sentences_dev, sentences_test)
# corpus = corpus.downsample(0.1)
print(corpus)

tag_type="NER"
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

charlm_embedding_forward = CharLMEmbeddings(model_path + "flair/embeddings/" + which_lm + "-forward/best-lm.pt")
charlm_embedding_backward = CharLMEmbeddings(model_path + "flair/embeddings/" + which_lm + "-backward/best-lm.pt")
# charlm_embedding_forward = CharLMEmbeddings("mix-forward")
# charlm_embedding_backward = CharLMEmbeddings("mix-backward")

print(charlm_embedding_forward.embedding_length)
print(charlm_embedding_backward.embedding_length)

embeddings = StackedEmbeddings(embeddings=[charlm_embedding_forward, charlm_embedding_backward])
# embeddings = StackedEmbeddings(embeddings=[charlm_embedding_backward])

print(embeddings.embedding_length)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

trainer: SequenceTaggerTrainer = SequenceTaggerTrainer(tagger, corpus, test_mode=False)

trainer.train(model_path + '/flair/seq-tagger/' + which_lm + '-seq-tagger',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=1000,
              train_with_dev=False)
