import random
import argparse
import os

def downsample(data_path, target_dir):
    filename = os.path.basename(data_path)
    print(filename)
    docs = []
    print(data_path)
    with open(data_path, "r") as f:
        temp_d = []
        for line in f:
            ar = line.replace("\n", "")
            if len(ar.split("\t")) == 2:
                temp_d.append(ar)
            else:
                docs.append(temp_d)
                temp_d = []

    print(len(docs))
    percentage = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]

    len_ds = []
    n_sents = []
    for perc in percentage:
        len_ds.append(len(docs))
        n_sents.append(round(len(docs) * perc))

    # i = 0
    data_write = []
    for i, (_, n_sent, perc) in enumerate(zip(len_ds, n_sents, percentage)):
        data_downsample = open(target_dir + "/"  + filename + str(perc) + "-" + str(n_sent) + ".conll", "w")

        if i == 0:
            n_sentence = n_sent
        else:
            n_sentence = n_sent - n_sents[i-1]

        print(len(docs), n_sent, n_sentence)
        if len(docs) > n_sentence:
            sampled_data = random.sample(docs, n_sentence)
        else:
            sampled_data = docs
        data_write += sampled_data
        docs = [x for x in docs if x not in sampled_data]

        print(len(data_write))
        for elem in data_write:
            for pair in elem:
                data_downsample.write(pair + "\n")
            data_downsample.write("\n")
        data_downsample.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_file', metavar='FILE', help='path to the corpus file')
    parser.add_argument('-o', '--output-dir', metavar='DIR', default=os.getcwd(),
                        help='output directory (default: {})'.format(os.getcwd()))
    # parser.add_argument('--encoding', default='utf-8',
    #                     help='file encoding (default: utf-8)')
    
    args = parser.parse_args()
    print(args)
    downsample(args.corpus_file, args.output_dir)


if __name__ == '__main__':
    main()
