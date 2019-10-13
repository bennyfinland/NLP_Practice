import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import tensorflow.contrib.rnn as rnn
import data_utils


class Model(object):
    def __init__(self, config):
        self.config = config
        self.lr = config['lr']
        self.word_dim = config['word_dim']
        self.lstm_dim = config['lstm_dim']
        self.seg_dim = config['seg_dim']
        self.num_tags = config['num_tags']
        self.num_words = config['num_words']
        self.num_sges = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # 申请占位符
        self.word_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="wordInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")

        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

        used = tf.sign(tf.abs(self.word_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.word_inputs)[0]
        self.num_setps = tf.shape(self.word_inputs)[-1]

        # embedding层单词和分词信息
        embedding = self.embedding_layer(self.word_inputs, self.seg_inputs, config)

        # lstm输入层
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)

        # lstm输出层
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)

        # 投影层
        self.logits = self.project_layer(lstm_outputs)

        # 损失
        self.loss = self.crf_loss_layer(self.logits, self.lengths)

        with tf.variable_scope('optimizer'):
            optimizer = self.config['optimizer']
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradDAOptimizer(self.lr)
            else:
                raise Exception("优化器错误")
            # 下面进行的是梯度的裁剪
            grad_vars = self.opt.compute_gradients(self.loss)
            capped_grad_vars = [[tf.clip_by_value(g, -self.config['clip'], self.config['clip']), v] for g, v in
                                grad_vars]

            self.train_op = self.opt.apply_gradients(capped_grad_vars, self.global_step)

            # 保存模型
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, word_inputs, seg_inputs, config, name=None):
        """
        :param word_inputs: one-hot编码.其实所有字的one_hot编码
        :param seg_inputs: 分词特征
        :param config: 配置
        :param name: 层的命名
        :return:  shape = [word_inputs,word_dim+seg_dim]
        """
        embedding = []
        with tf.variable_scope("word_embedding" if not name else name), tf.device('/cpu:0'):
            self.word_lookup = tf.get_variable(
                name="word_embedding",
                shape=[self.num_words, self.word_dim],
                initializer=self.initializer
            )
            embedding.append(tf.nn.embedding_lookup(self.word_lookup, word_inputs))

            if config['seg_dim']:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_sges, self.seg_dim],
                        initializer=self.initializer
                    )
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        return embed

    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :param lstm_dim:
        :param name:
        :return: [batch_size, num_steps, 2*lstm_dim]
        为何返回是2*lstm_dim，因为其是双向的lstm。每个方向的输出为lstm_dim
        """
        with tf.variable_scope("word_biLSTM" if not name else name):
            lstm_cell = {}
            for direction in ['forward', 'backward']:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True
                    )
            outputs, final_status = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell['forward'],
                lstm_cell['backward'],
                lstm_inputs,
                dtype=tf.float32,

                sequence_length=lengths
            )
        # 因为单向的lstm输出的格式为[batch_size, num_steps,lstm_dim]。
        # 2表示在lstm_dim这个维度进行拼接。
        # 个人觉得outputs的输出格式为[[batch_size, num_steps,lstm_dim],[batch_size, num_steps,lstm_dim]]
        # 即是一个list。list里面的每一个元素是单向的lstm的输出
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        个人觉得lstm_outputs: [batch_size, num_steps, lstm_dim * 2]  num_steps表示每个句子里面字的数量。即每个批次的句子长度
        :param name:
        :return: [btch_size,num_steps, num_tags]
        """
        with tf.variable_scope('project_layer' if not name else name):
            with tf.variable_scope('hidden_layer'):
                W = tf.get_variable(
                    "W",
                    shape=[self.lstm_dim * 2, self.lstm_dim],
                    dtype=tf.float32,
                    initializer=self.initializer
                )
                b = tf.get_variable(
                    "b",
                    shape=[self.lstm_dim],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer()
                )
                out_put = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])  # 得到所有的字，将所有的字最后编码为lstm_dim长度
                hidden = tf.tanh(tf.nn.xw_plus_b(out_put, W, b))

            with tf.variable_scope('logits'):
                W = tf.get_variable(
                    "W",
                    shape=[self.lstm_dim, self.num_tags],
                    dtype=tf.float32,
                    initializer=self.initializer
                )
                b = tf.get_variable(
                    "b",
                    shape=[self.num_tags],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer()
                )
                # 最后将每个字编码为num_tags。即最后想要得到每个字属于每个tag的概率
                pred = tf.nn.xw_plus_b(hidden, W, b)
        #  返回原始的shape。即batch_size,num_setps,num_tags
        return tf.reshape(pred, [-1, self.num_setps, self.num_tags])

    def crf_loss_layer(self, project_logits, lenghts, name=None):
        """
        # 个人觉得是[-1, self.num_setps, self.num_tags]
        :param project_logits: [1, num_steps, num_tages]
        :param lenghts:
        :param name:
        :return: scalar loss
        听说下面是固定的写法
        """
        with tf.variable_scope('crf_loss' if not name else name):
            small_value = -10000.0
            # 下面是对于一个字。但是最后一维，比原来的标签长度多了一个元素
            start_logits = tf.concat(
                [
                    small_value * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                    tf.zeros(shape=[self.batch_size, 1, 1])
                ],
                axis=-1
            )

            pad_logits = tf.cast(
                small_value *
                tf.ones(shape=[self.batch_size, self.num_setps, 1]),
                dtype=tf.float32
            )

            # 貌似是在列的位置最后拼接一个元素.所以此时project_layer层输出的每个字最后一层多了一个元素
            # 即在最后一个维度填充了一个元素
            logits = tf.concat(
                [project_logits, pad_logits],
                axis=-1
            )
            # 此时相当于在每个批次的，每个句子开头位置添加了一个字
            logits = tf.concat(
                [start_logits, logits],
                axis=1
            )
            # 因为self.targets.shape = [batch_size,num_steps].所以下面的操作，类似于在每个句子前面添加了一个字
            # 所以此时就和上面的填充的形状tf.concat([start_logits, logits],axis=1)
            # 对应了起来
            targets = tf.concat(
                [tf.cast(
                    self.num_tags * tf.ones([self.batch_size, 1]),
                    tf.int32
                ),
                    self.targets
                ]
                ,
                axis=-1
            )
            # 每个状态之间的转移矩阵
            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer
            )

            log_likehood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lenghts + 1  # 因为上面在句子的开头位置添加了一个字
            )
            return tf.reduce_mean(-log_likehood)

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size,num_steps, num_tags]
        :param lengths:
        :param matrix: 概率转移矩阵
        :return:
        """
        paths = []
        small = -1000.0
        # start类似start = np.asarray([[3] * 4 + [0]]) = [[3 3 3 3 0]]
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train:
        :param batch: 批次数据
        :return:
        """
        _, words, segs, tags = batch
        feed_dict = {
            self.word_inputs: np.asarray(words),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0
        }

        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config['dropout_keep']
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess:
        :param is_train:
        :param bath:
        :return:
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op], feed_dict
            )
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess:
        :param data_manager:
        :param id_to_tag:
        :return:
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, logits = self.run_step(sess, False, batch)
            batch_paths = self.decode(logits, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = data_utils.bioes_to_bio([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = data_utils.bioes_to_bio([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results
