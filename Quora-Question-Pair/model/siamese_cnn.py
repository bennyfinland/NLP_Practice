import copy
import datetime
import logging
import sys
import tensorflow as tf

class SiameseCNN(object):
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
            self._m_ph_sent2 = tf.placeholder(tf.int32, [None, None], name="sent2")
            self._m_ph_label = tf.placeholder(tf.float32, [None, 2], name="label")
            self._m_ph_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

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

            dropout_embedded_sent1 = tf.nn.dropout(embedded_sent1, keep_prob=self._m_ph_keep_prob)
            dropout_embedded_sent2 = tf.nn.dropout(embedded_sent2, keep_prob=self._m_ph_keep_prob)

        with tf.name_scope('sentence_features'):
            sent1_features = self._build_conv_features(dropout_embedded_sent1)
            sent2_features = self._build_conv_features(dropout_embedded_sent2)
            #dropout_sent1_features = tf.nn.dropout(sent1_features, keep_prob=self._m_ph_keep_prob)
            #dropout_sent2_features = tf.nn.dropout(sent2_features, keep_prob=self._m_ph_keep_prob)
            dropout_sent1_features = tf.identity(sent1_features)
            dropout_sent2_features = tf.identity(sent2_features)

        with tf.name_scope("feature_mapping"):
            sent_diff = dropout_sent1_features - dropout_sent2_features
            sent_mul = tf.multiply(dropout_sent1_features, dropout_sent2_features)
            features = tf.concat([sent_diff, sent_mul, dropout_sent1_features, dropout_sent2_features], axis=1)
            dropout_features = tf.nn.dropout(features, keep_prob=self._m_ph_keep_prob)

            cnn_feature_num = self._m_config["num_filters"] * len(self._m_config["filter_sizes"])
            W = tf.Variable(tf.truncated_normal(
                            shape=[cnn_feature_num * 4, self._m_config["label_num"]],
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

    def _build_conv_features(self, sentence_matrix):
        sentence_matrix = tf.expand_dims(sentence_matrix, axis=-1)
        conv_features = []
        for i, filter_size in enumerate(self._m_config["filter_sizes"]):
            with tf.variable_scope("conv-maxpool-%s" % i, reuse=tf.AUTO_REUSE):
                filter_shape = [
                    filter_size,
                    self._m_config["embedding_dim"],
                    1,
                    self._m_config["num_filters"]
                ]
                cnn_weights = tf.get_variable(
                    name="cnn_weights",
                    shape=filter_shape,
                    initializer=tf.contrib.layers.xavier_initializer(),
                )

                # [batch_size, sequence_length - filter_size + 1, 1, num_filters]
                sentence_conv = tf.nn.conv2d(
                    sentence_matrix,
                    cnn_weights,
                    strides=[1, 1, 1, 1],
                    padding="VALID"
                )
                # [batch_size, 1, 1, num_filters]
                sentence_maxpool = tf.reduce_max(sentence_conv, axis=1, keepdims=True)
                sentence_squeeze = tf.squeeze(sentence_maxpool, axis=[1, 2])
                conv_features.append(tf.nn.relu(sentence_squeeze))
        return tf.concat(conv_features, axis=1)

    def _train_one_batch(self, sent1, sent1_size, sent2, sent2_size, label):
        feed_dict = {
            self._m_ph_sent1: sent1,
            self._m_ph_sent2: sent2,
            self._m_ph_label: label,
            self._m_ph_keep_prob: self._m_config["keep_prob"],
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

            if step % 500 == 0:
                print(step, self.evaluate(dev_data))
                sys.stdout.flush()

    def evaluate(self, data):
        data.reinitialize()
        correct_samples = total_samples = 0
        for batch_data in data:
            sent1, sent1_size, sent2, sent2_size, label = batch_data

            feed_dict = {
                self._m_ph_sent1: sent1,
                self._m_ph_sent2: sent2,
                self._m_ph_label: label,
                self._m_ph_keep_prob: 1.0,
            }
            accuracy = self._m_session.run(self._m_accuracy, feed_dict=feed_dict)
            total_samples += sent1.shape[0]
            correct_samples += sent1.shape[0] * accuracy
        return correct_samples / total_samples

