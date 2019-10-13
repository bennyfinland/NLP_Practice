from dataset import Dataset

import model
import argparse
import tensorflow as tf


def main():
  args = parse_args()

  with tf.Session() as sess:
    word_dict = tf.contrib.lookup.index_table_from_file(
      args.vocab_fname,
      num_oov_buckets=2,
      key_column_index=0,
      delimiter='\t',
      vocab_size=10000,
    )
    sess.run(tf.tables_initializer())

    model_config = {
      "vocab_size": sess.run(word_dict.size()),
      "embedding_dim": 50,
      "label_num": 2,
      "learning_rate": 3e-4,
      "lstm_dim": 100,
      "filter_sizes": [2, 3, 4],
      "num_filters": 32,
      "keep_prob": 0.8,
      "cnn_block": 2,
      "max_sequence_length": 25,

      "l2_coef": 0.01,
      "projection": {
        "enabled": True,
        "dim": 50,
      },
      "attention": {
        "dim": 100,
      },
      "compare": {
        "dim": 100,
      },
      "aggregate": {
        "dim": [50],
      },
    }

    print("Initialize model")
    #m = model.SiameseLSTM(sess, model_config).initialize()
    m = model.SiameseCNN(sess, model_config).initialize()
    #m = model.MatchPyramid(sess, model_config).initialize()
    #m = model.Decomposable(sess, model_config).initialize()

    print("Training model")
    if isinstance(m, model.MatchPyramid):
      max_sequence_length = model_config["max_sequence_length"]
    else:
      max_sequence_length = None

    train_data = Dataset(sess, args.train_fname, word_dict,
      repeat_count=15, shuffle=True, max_sequence_length=max_sequence_length)

    dev_data = Dataset(sess, args.dev_fname, word_dict, max_sequence_length=max_sequence_length)
    test_data = Dataset(sess, args.test_fname, word_dict, max_sequence_length=max_sequence_length)

    m.train(train_data, dev_data)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--vocab_fname", default="./data/train_test/vocab.txt",
                     help="Vocabulary file name")

  parser.add_argument("--train_fname", default="./data/train_test/train.csv",
                     help="Training data file name")

  parser.add_argument("--dev_fname", default="./data/train_test/dev.csv",
                     help="Development data file name")

  parser.add_argument("--test_fname", default="./data/train_test/test.csv",
                    help="Testing data file name")
  return parser.parse_args()


if __name__ == "__main__":
  exit(main())
