import tensorflow as tf


class TensorBoardWriter(object):
    def __init__(self, logdir):
        self._writer = tf.summary.FileWriter(logdir)

    def add(self, step, key, value):
        summary = tf.summary.Summary()
        summary_value = summary.value.add()
        summary_value.tag = key
        summary_value.simple_value = value
        self._writer.add_summary(summary, global_step=step)

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()
