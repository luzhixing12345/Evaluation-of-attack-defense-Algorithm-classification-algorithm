import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def dataset_to_iterator(dataset, session):
    ''' A simple wrapper to get dataset in numpy format from the graph-mode dataset. '''
    iterator = dataset.make_one_shot_iterator().get_next()
    while True:
        try:
            yield session.run(iterator)
        except tf.errors.OutOfRangeError:
            return
