"""Decomposable 模型示例 by 解惑者
"""
import copy
import datetime
import logging
import numpy as np
import tensorflow as tf

class Decomposable(object):
    def __init__(self, session, model_config):
        """model_config should contains these configure
        {
            "l2_coef": 0.01,
            "projection": {
                "enabled": True,
                "dim": 50,
            },
            "attention": {
                "dim": 60,
            },
            "compare": {
                "dim": 70,
            },
            "aggregate": {
                "dim": [70, 30],
            },
        }
        """
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

            self._m_ph_learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
            self._m_ph_dropout_rate = tf.placeholder(tf.float32, shape=(), name='dropout')
            self._m_ph_clip_norm_value = tf.placeholder(tf.float32, shape=(), name='clip_norm_value')

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

        with tf.variable_scope("embedding_projection", reuse=tf.AUTO_REUSE):
            # project dimension of input embeddings into another dimension
            projected1, projected_dim = self._project_embeddings(embedded_sent1)
            projected2, _ = self._project_embeddings(embedded_sent2)

        # the architecture has 3 main steps: soft align, compare and aggregate
        with tf.name_scope("inter-attention"):
            # alpha and beta have shape (batch, time_steps, embeddings)
            alpha, beta = self._attend(projected1, projected2)

        with tf.variable_scope("compare", reuse=tf.AUTO_REUSE):
            v1 = self._compare(projected1, beta, self._m_ph_sent1_size)
            v2 = self._compare(projected2, alpha, self._m_ph_sent2_size)

        with tf.variable_scope("aggregate"):
            aggregate_features = self._aggregate(v1, v2)

        with tf.variable_scope('logits'):
            pre_logits_dim = self._m_config['aggregate']['dim'][-1]
            weights = tf.get_variable(
                'weights',
                shape=[pre_logits_dim, self._m_config["label_num"]],
                initializer=tf.random_normal_initializer(0.0, 0.1)
            )
            bias = tf.get_variable(
                'bias',
                shape=[self._m_config["label_num"]],
                initializer=tf.zeros_initializer(),
            )
            self._m_logits = tf.nn.xw_plus_b(aggregate_features, weights, bias)

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels=self._m_ph_label, logits=self._m_logits)

            weights = [w for w in tf.trainable_variables() if 'kernel' in w.name]
            l2_partial_sum = tf.reduce_sum([tf.nn.l2_loss(weight) for weight in weights])
            l2_loss = tf.multiply(self._m_config["l2_coef"], l2_partial_sum, name='l2_loss')
            self._m_loss = tf.add(tf.reduce_mean(cross_entropy), l2_loss, name='loss')

        with tf.name_scope("accuracy"):
            self._m_prediction = tf.argmax(self._m_logits, axis=1)
            correct = tf.equal(self._m_prediction, tf.argmax(self._m_ph_label, axis=1))
            self._m_accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.name_scope("optimizer"):
            self._m_global_step = tf.Variable(0, name="global_step", trainable=False)
            self._m_optimizer = tf.train.AdamOptimizer(self._m_config["learning_rate"])
            self._m_train_op = self._m_optimizer.minimize(
                                    self._m_loss, global_step=self._m_global_step)

    def _project_embeddings(self, embeddings):
        if not self._m_config['projection']['enabled']:
            return embeddings, self._m_config['embedding_dim']

        projection_dim = self._m_config['projection']['dim']
        embeddings_2d = tf.reshape(embeddings, [-1, self._m_config["embedding_dim"]])
        weights = tf.get_variable(
            'weights',
            shape=[self._m_config["embedding_dim"], projection_dim],
            initializer=tf.random_normal_initializer(0.0, 0.1)
        )
        projected = tf.matmul(embeddings_2d, weights)

        time_steps = tf.shape(embeddings)[1]
        projected_3d = tf.reshape(projected, [-1, time_steps, projection_dim])
        return projected_3d, projection_dim

    def _attend(self, sent1, sent2):
        # this is F in the paper
        # (batch, time_steps, num_units)
        with tf.variable_scope("sentence_projection", reuse=tf.AUTO_REUSE):
            sent1_proj = self._feedforward(sent1, self._m_config["attention"]["dim"])
            sent2_proj = self._feedforward(sent2, self._m_config["attention"]["dim"])

        # compute the unnormalized attention for all word pairs
        # raw_attentions has shape (batch, time_steps1, time_steps2)
        raw_attentions = tf.matmul(sent1_proj, sent2_proj, transpose_b=True)

        # now get the attention softmaxes
        masked1 = self._mask3d(raw_attentions, self._m_ph_sent2_size, -np.inf)
        att_sent1 = tf.nn.softmax(masked1)

        att_transposed = tf.transpose(raw_attentions, [0, 2, 1])
        masked2 = self._mask3d(att_transposed, self._m_ph_sent1_size, -np.inf)
        att_sent2 = tf.nn.softmax(masked2)

        alpha = tf.matmul(att_sent2, sent1, name='alpha')
        beta = tf.matmul(att_sent1, sent2, name='beta')
        return alpha, beta

    def _compare(self, sentence, attention, sentence_length):
        # sent_and_alignment has shape [batch, time_steps, num_units]
        inputs = [sentence, attention, sentence - attention, sentence * attention]
        features = tf.concat(axis=2, values=inputs)
        return self._feedforward(features, self._m_config['compare']['dim'])

    def _aggregate(self, v1, v2):
        # sum over time steps; resulting shape is [batch, num_units]
        v1 = self._mask3d(v1, self._m_ph_sent1_size, 0, 1)
        v2 = self._mask3d(v2, self._m_ph_sent2_size, 0, 1)
        v1_sum = tf.reduce_sum(v1, 1)
        v2_sum = tf.reduce_sum(v2, 1)
        v1_max = tf.reduce_max(v1, 1)
        v2_max = tf.reduce_max(v2, 1)
        aggregate_features = tf.concat([v1_sum, v2_sum, v1_max, v2_max], axis=1)
        return self._feedforward(aggregate_features, self._m_config['aggregate']['dim'])

    def _feedforward(self, inputs, hidden_units):
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]

        last_inputs = inputs
        for i, dim in enumerate(hidden_units):
            with tf.variable_scope('layer_%d' % (i)):
                inputs = tf.nn.dropout(last_inputs, self._m_ph_dropout_rate)
                last_inputs = tf.layers.dense(inputs, dim, tf.nn.relu)
        return last_inputs

    def _mask3d(self, values, sentence_sizes, mask_value, axis=2):
        if axis != 1 and axis != 2:
            raise ValueError("'axis' must be 1 or 2")

        if axis == 1:
            values = tf.transpose(values, [0, 2, 1])

        time_steps1 = tf.shape(values)[1]
        time_steps2 = tf.shape(values)[2]

        # [batch_size, time_step1, time_step2]
        ones = tf.ones_like(values, dtype=tf.float32)
        pad_values = mask_value * ones
        # [batch_size, time_step2]
        mask = tf.sequence_mask(sentence_sizes, time_steps2)

        # [batch_size, 1, time_step2]
        mask3d = tf.expand_dims(mask, 1)
        # [batch_size, time_step1, time_step2]
        mask3d = tf.tile(mask3d, (1, time_steps1, 1))

        masked = tf.where(mask3d, values, pad_values)
        if axis == 1:
            masked = tf.transpose(masked, [0, 2, 1])
        return masked

    def _train_one_batch(self, sent1, sent1_size, sent2, sent2_size, label):
        feed_dict = {
            self._m_ph_sent1: sent1,
            self._m_ph_sent1_size: sent1_size,
            self._m_ph_sent2: sent2,
            self._m_ph_sent2_size: sent2_size,
            self._m_ph_label: label,
            self._m_ph_dropout_rate: self._m_config["keep_prob"], 
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
                self._m_ph_dropout_rate: 1.0,
            }
            accuracy = self._m_session.run(self._m_accuracy, feed_dict=feed_dict)
            total_samples += sent1.shape[0]
            correct_samples += sent1.shape[0] * accuracy
        return correct_samples / total_samples

