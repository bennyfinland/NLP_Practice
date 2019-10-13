import tensorflow as tf


class Dataset(object):
  def __init__(self, session, data_fname, word2idx, repeat_count=1, batch_size=32,
                            shuffle=False, buffer_size=10000, max_sequence_length=None):
    self._m_session = session
    self._m_dataset = self._build_dataset(data_fname, word2idx, repeat_count, batch_size,
                        shuffle, buffer_size, max_sequence_length)
    self._m_iterator = self._m_dataset.make_initializable_iterator()
    self._m_next = self._m_iterator.get_next()

  def reinitialize(self):
    self._m_session.run(self._m_iterator.initializer)

  def _build_dataset(self, data_fname, word2idx, repeat_count, batch_size, shuffle,
                           buffer_size, max_sequence_length):
    def process_line(line):
      parsed_line = tf.decode_csv(line, ["", "", 0])
      sent1 = word2idx.lookup(tf.string_split([parsed_line[0]], ' ').values)
      sent2 = word2idx.lookup(tf.string_split([parsed_line[1]], ' ').values)
      label = tf.one_hot(parsed_line[2], 2)
      del parsed_line
      if isinstance(max_sequence_length, int):
        sent1 = sent1[: max_sequence_length]
        sent2 = sent2[: max_sequence_length]
      return [sent1, tf.size(sent1), sent2, tf.size(sent2), label]

    dataset = tf.data.TextLineDataset(data_fname).skip(1).map(process_line)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=buffer_size)

    id_pad_word = word2idx.lookup(tf.constant('<pad>'))
    padding_values = (id_pad_word, 0, id_pad_word, 0, 0.0)
    padded_shapes = (
      tf.TensorShape([max_sequence_length]),
      tf.TensorShape([]),
      tf.TensorShape([max_sequence_length]),
      tf.TensorShape([]),
      tf.TensorShape([2])
    )
    dataset = dataset.repeat(repeat_count).padded_batch(
      batch_size,
      padded_shapes=padded_shapes,
      padding_values=padding_values
    )
    return dataset

  def __iter__(self):
    while True:
      try:
        yield self._m_session.run(self._m_next)
      except tf.errors.OutOfRangeError as e:
        break
