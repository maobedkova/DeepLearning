#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
# Import `eager` from `contrib` module, into `tfe`
import tensorflow.contrib.eager as tfe

# Enable eager mode using `enable_eager_execution`.
tfe.enable_eager_execution()

class Network(tfe.Network):
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, args):
        super(Network, self).__init__(name='')

        # Add layers to the Network, using `self.track_layer`.
        # Create:
        # - a Conv2D layer with 8 filters, kernel size 3 and ReLU activation
        self.conv_8 = self.track_layer(tf.layers.Conv2D(8, 3, activation=tf.nn.relu))

        # - a Conv2D layer with 16 filters, kernel size 3 and ReLU activation
        self.conv_16 = self.track_layer(tf.layers.Conv2D(16, 3, activation=tf.nn.relu))

        # - a MaxPooling2D layer with pooling size 2 and stride 2
        self.max_pool = self.track_layer(tf.layers.MaxPooling2D(2, 2))

        # - a Dense layer with 256 neurons and ReLU activation
        self.dense_256 = self.track_layer(tf.layers.Dense(256, activation=tf.nn.relu))

        # - a Dense layer with self.LABELS neurons and no activation
        self.dense_LABELS = self.track_layer(tf.layers.Dense(self.LABELS, activation=None))

        # - a Dropout layer with 0.5 dropout rate (without explicit seed)
        self.dropout = self.track_layer(tf.layers.Dropout(0.5))

        # - a Flatten layer
        self.flatten = self.track_layer(tf.layers.Flatten())

        self.global_step = tf.train.create_global_step()
        self.optimizer = tf.train.AdamOptimizer()
        self.summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

    def call(self, inputs, training):
        # Process inputs, using the above created layers:
        # - convolution with 8 filters
        hidden = self.conv_8(inputs)
        # - max pooling
        hidden = self.max_pool(hidden)
        # - convolution with 16 filters
        hidden = self.conv_16(hidden)
        # - max pooling
        hidden = self.max_pool(hidden)
        # - flattening layer
        hidden = self.flatten(hidden)
        # - dense layer with 256 neurons
        hidden = self.dense_256(hidden)
        # - dropout layer, utilizing `training` parameter
        hidden = self.dropout(hidden, training)
        # - dense layer with self.LABELS neurons
        logits = self.dense_LABELS(hidden)
        # Return the computed logits.
        return logits

    def predict(self, logits):
        return tf.argmax(logits, axis=1)

    def train_epoch(self, dataset):
        # Iterate over images and labels using `tfe.Iterator` in the `dataset`.
        for images, labels in tfe.Iterator(dataset):
            # Use `tfe.GradientTape` to store loss computation
            with tfe.GradientTape() as tape:
                # Compute `logits` using `self(images, training=True)`
                logits = self.call(images, training=True)

                # Compute `loss` using `tf.losses.sparse_softmax_cross_entropy`
                loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

            # Compute `gradients` of `loss` with respect to `self.variables` using the `GradientTape`
            gradients = tape.gradient(loss, self.variables)

            with self.summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                tf.contrib.summary.scalar("train/loss", loss)
                accuracy = tfe.metrics.Accuracy("train/accuracy"); accuracy(labels, self.predict(logits)); accuracy.result()

            self.optimizer.apply_gradients(zip(gradients, self.variables), global_step=self.global_step)

    def evaluate(self, dataset_name, dataset):
        # Create `accuracy` metric using `tfe.metrics.Accuracy` with `dataset_name + "/accuracy"` name
        accuracy = tfe.metrics.Accuracy(dataset_name + "/accuracy")

        # Iterate over images and labels using `tfe.Iterator` in the `dataset`.
        for images, labels in tfe.Iterator(dataset):
            # Compute `logits` using `self(images, training=False)`
            logits = self.call(images, training=False)

            # Update accuracy metric using the `labels` and predictions from `logits`
            accuracy(labels, self.predict(logits))

        with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            # This both adds a summary and returns the result
            return accuracy.result()


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)
    tf.set_random_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(".", reshape=False, seed=42)

    # Create `train` dataset using `from_tensor_slices`, shuffle it
    # (without explicit seed) using a buffer of 60000 and generate batches of
    # size args.batch_size. Note that `mnist.train.labels` must be converted to
    # `np.int64` (for example using mnist.train.labels.astype(np.int64) call).
    train_labels = mnist.train.labels.astype(np.int64)
    train = tf.data.Dataset.from_tensor_slices((mnist.train.images, train_labels))
    train = train.shuffle(60000)
    train = train.batch(args.batch_size)

    # Create `dev` and `test` datasets similarly, but without shuffling
    # and using `mnist.validation` and `mnist.test`, respectively.
    test_labels = mnist.test.labels.astype(np.int64)
    test = tf.data.Dataset.from_tensor_slices((mnist.test.images, test_labels))
    test = test.batch(args.batch_size)
    dev_labels = mnist.validation.labels.astype(np.int64)
    dev = tf.data.Dataset.from_tensor_slices((mnist.validation.images, dev_labels))
    dev = dev.batch(args.batch_size)

    # Construct the network
    network = Network(args)

    # Train
    for i in range(args.epochs):
        network.train_epoch(train)

        print("Dev acc after epoch {}: {}".format(i + 1, network.evaluate("dev", dev)))
    print("Test acc: {}".format(network.evaluate("test", test)))
