import os
import math
import csv
import argparse
import collections
import itertools
import operator
import logging
import pandas as pd
import numpy as np
import nltk


def split_train_test(data_fname, save_path, dev_ratio=0.02, test_ratio=0.1):
    logging.info("split original data into train/dev/test")
    all_data = pd.read_csv(data_fname)
    sample_idx = np.array(all_data.index)
    np.random.shuffle(sample_idx)

    dev_sample_num = math.ceil(sample_idx.shape[0] * dev_ratio)
    test_sample_num = math.ceil(sample_idx.shape[0] * test_ratio)
    train_sample_num = sample_idx.shape[0] - dev_sample_num - test_sample_num

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cur_pos = 0
    data = {}
    for sample_size, fname in [[train_sample_num, 'train.csv'],
                               [dev_sample_num, 'dev.csv'],
                               [test_sample_num, 'test.csv']]:
        all_data.iloc[sample_idx[cur_pos: cur_pos + sample_size]].to_csv(
            os.path.join(save_path, fname),
            index=False,
            quoting=csv.QUOTE_ALL
        )
        cur_pos += sample_size
        data[fname.split('.')[0]] = os.path.join(save_path, fname)
    return data


def preprocess(data, save_path):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    res = {}
    for data_type, data_fname in data.items():
        logging.info("Processing %s data" % (data_type))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        samples = []
        for idx, row in pd.read_csv(data_fname).fillna("").iterrows():
            sent1 = ' '.join(tokenizer.tokenize(row['question1'].lower()))
            sent2 = ' '.join(tokenizer.tokenize(row['question2'].lower()))
            samples.append((sent1, sent2, row['is_duplicate']))
            if idx % 10000 == 0:
                logging.info("Processed %.3fK lines" % ((idx + 1) / 1000))
        logging.info("Done %.2fK lines" % ((idx + 1) / 1000))
        pd.DataFrame(samples, columns=['question1', 'question2', 'is_duplicate']).to_csv(
            os.path.join(save_path, os.path.basename(data_fname)),
            index=False,
            quoting=csv.QUOTE_ALL
        )
        res[data_type] = os.path.join(save_path, os.path.basename(data_fname))
    return res


def extract_vocabulary(train_fname, save_path):
    logging.info("Extract vocabulary")
    docs = []
    for idx, row in pd.read_csv(train_fname).fillna("").iterrows():
        docs.append(row['question1'].split())
        docs.append(row['question2'].split())

    word_cnt = collections.Counter(itertools.chain(*docs))
    with open(os.path.join(save_path, "vocab.txt"), 'w') as fv:
        fv.write("<pad>\t-1\n")
        for word, cnt in sorted(word_cnt.items(), key=operator.itemgetter(1), reverse=True):
            fv.write("{}\t{}\n".format(word, cnt))
    return word_cnt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fname", default="../data/origin_train.csv.zip",
                        help="Kaggle train zip data")

    parser.add_argument("-o", "--output_path", default="../data",
                        help="Output saving path")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    np.random.seed(9507)

    args = parse_args()
    save_path = args.output_path

    # step 1. split origin data into train/dev/test datasets.
    data = split_train_test(args.data_fname, os.path.join(save_path, 'origin_split'))

    # step 2. preprocess train/dev/test datasets.
    data = preprocess(data, os.path.join(save_path, 'train_test'))

    # step 3. extract vocabulary.
    extract_vocabulary(data['train'], os.path.join(args.output_path, 'train_test'))

    return 0


if __name__ == "__main__":
    exit(main())

