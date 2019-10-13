"""MatchPyramid 模型示例 by 解惑者
"""
import sys
import copy
import datetime
import logging
import tensorflow as tf


class MatchPyramid(object):
    def __init__(self, session, model_config):
        self._m_session = session
        self._m_config = copy.deepcopy(model_config)
        self._build_graph()

    def initialize(self):
        self._m_session.run(tf.global_variables_initializer())
        return self

    def _build_graph(self):
        with tf.name_scope("input"):
            self._m_ph_sent1 = tf.placeholder(
                tf.int32,
                [None, self._m_config["max_sequence_length"]],
                name="sent1"
            )
            self._m_ph_sent2 = tf.placeholder(
                tf.int32,
                [None, self._m_config["max_sequence_length"]],
                name="sent2"
            )
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

            # 构建相似性矩阵，并且使用CNN对齐分类
            # sent.shape = [batch_size, sequence_length, dim]
            picture = tf.matmul(dropout_embedded_sent1, dropout_embedded_sent2, transpose_b=True)
            self._m_picture = tf.expand_dims(picture, axis=-1)

        pooled_outputs = []
        for i, filter_size in enumerate(self._m_config["filter_sizes"]):
            with tf.name_scope("conv-max-pool-%s" % filter_size):
                filter_shape = [filter_size, filter_size, 1, 1]
                conv_weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

                conv_features = tf.nn.conv2d(
                    input=self._m_picture,
                    filter=conv_weight,
                    strides=[1, 1, 1, 1],
                    padding="SAME")

                maxpool_features = tf.nn.max_pool(
                    value=conv_features,
                    ksize=[1, 4, 4, 1],
                    strides=[1, 4, 4, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(tf.layers.flatten(tf.squeeze(maxpool_features, axis=3)))

        self._m_cnn_features = tf.concat(pooled_outputs, 1)
        self._m_cnn_features_dropout = tf.nn.dropout(self._m_cnn_features, self._m_ph_keep_prob)

        with tf.name_scope("full_connected_layer"):
            feature_size = self._m_cnn_features_dropout.shape.as_list()[1]
            W = tf.get_variable(
                name="W",
                shape=[feature_size, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            self._m_logits = tf.nn.xw_plus_b(self._m_cnn_features_dropout, W, b)

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

