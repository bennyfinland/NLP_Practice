import copy
import datetime
import logging
import tensorflow as tf

class SiameseLSTM(object):
    def __init__(self, session, model_config):
        self._m_session = session
        self._m_config = copy.deepcopy(model_config)
        self._build_graph()

    def initialize(self):
        self._m_session.run(tf.global_variables_initializer())
        return self

    def _build_graph(self):
        with tf.name_scope("input"):
            self._m_ph_sent1 = tf.placeholder(tf.int32, [None, None], name="sent1")
            self._m_ph_sent1_size = tf.placeholder(tf.int32, [None], name="sent1_size")

            self._m_ph_sent2 = tf.placeholder(tf.int32, [None, None], name="sent2")
            self._m_ph_sent2_size = tf.placeholder(tf.int32, [None], name="sent2_size")

            self._m_ph_label = tf.placeholder(tf.float32, [None, 2], name="label")

        with tf.name_scope('embeddings'):
            self._m_token_embeddings = tf.Variable(
                tf.truncated_normal(
                    [self._m_config["vocab_size"], self._m_config["embedding_dim"]],
                    stddev=0.1
                ),
                name="token_embeddings"
            )
            embedded_sent1 = tf.nn.embedding_lookup(self._m_token_embeddings, self._m_ph_sent1)
            embedded_sent2 = tf.nn.embedding_lookup(self._m_token_embeddings, self._m_ph_sent2)
        self._m_embedded_sent1 = embedded_sent1

        with tf.name_scope('lstm_layer'):
            cell1 = tf.nn.rnn_cell.LSTMCell(
                self._m_config["lstm_dim"],
                state_is_tuple=True,
                reuse=tf.AUTO_REUSE
            )
            cell2 = tf.nn.rnn_cell.LSTMCell(
                self._m_config["lstm_dim"],
                state_is_tuple=True,
                reuse=tf.AUTO_REUSE
            )
            _, (_, output_cell1) = tf.nn.dynamic_rnn(
                cell1, embedded_sent1, dtype=tf.float32, sequence_length=self._m_ph_sent1_size)
            _, (_, output_cell2) = tf.nn.dynamic_rnn(
                cell1, embedded_sent2, dtype=tf.float32, sequence_length=self._m_ph_sent2_size)

        with tf.name_scope("feature_mapping"):
            sent_diff = output_cell1 - output_cell2
            sent_mul = tf.multiply(output_cell1, output_cell2)
            features = tf.concat([sent_diff, sent_mul, output_cell1, output_cell2], axis=1)

            W = tf.Variable(tf.truncated_normal(
                            shape=[self._m_config["lstm_dim"] * 4, self._m_config["label_num"]],
                            stddev=0.1, mean=0.0))
            b = tf.Variable(tf.truncated_normal(
                            shape=[self._m_config["label_num"]], stddev=0.1, mean=0.0))
            self._m_logits = tf.nn.xw_plus_b(features, W, b)

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels=self._m_ph_label, logits=self._m_logits)
            self._m_loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("accuracy"):
            self._m_prediction = tf.argmax(self._m_logits, axis=1)
            correct = tf.equal(self._m_prediction, tf.argmax(self._m_ph_label, axis=1))
            self._m_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.name_scope("optimizer"):
            self._m_global_step = tf.Variable(0, name="global_step", trainable=False)
            self._m_optimizer = tf.train.AdamOptimizer(self._m_config["learning_rate"])
            self._m_train_op = self._m_optimizer.minimize(
                                    self._m_loss, global_step=self._m_global_step)

    def _train_one_batch(self, sent1, sent1_size, sent2, sent2_size, label):
        feed_dict = {
            self._m_ph_sent1: sent1,
            self._m_ph_sent1_size: sent1_size,
            self._m_ph_sent2: sent2,
            self._m_ph_sent2_size: sent2_size,
            self._m_ph_label: label,
        }
        _, step, loss, accuracy = self._m_session.run(
            fetches=[self._m_train_op, self._m_global_step, self._m_loss, self._m_accuracy],
            feed_dict=feed_dict
        )
        time_str = datetime.datetime.now().isoformat()
        logging.info("{}: step {}, loss {:g}, accuracy {}".format(time_str, step, loss, accuracy))
        return step

    def train(self, train_data, dev_data):
        train_data.reinitialize()
        for train_batch_data in train_data:
            step = self._train_one_batch(*train_batch_data)

            if step % 1000 == 0:
                print(step, self.evaluate(dev_data))

    def evaluate(self, data):
        data.reinitialize()
        correct_samples = total_samples = 0
        for batch_data in data:
            sent1, sent1_size, sent2, sent2_size, label = batch_data

            feed_dict = {
                self._m_ph_sent1: sent1,
                self._m_ph_sent1_size: sent1_size,
                self._m_ph_sent2: sent2,
                self._m_ph_sent2_size: sent2_size,
                self._m_ph_label: label,
            }
            accuracy = self._m_session.run(self._m_accuracy, feed_dict=feed_dict)
            total_samples += sent1.shape[0]
            correct_samples += sent1.shape[0] * accuracy
        return correct_samples / total_samples

