#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # I
            features = tf.layers.conv2d(inputs=self.images,
                                        filters=32,
                                        strides=[2, 2],
                                        kernel_size=[10, 10],
                                        padding="same",
                                        activation=tf.nn.relu)

            features = tf.nn.relu(tf.layers.batch_normalization(features, training=self.is_training))

            features = tf.layers.max_pooling2d(inputs=features,
                                               pool_size=[2, 2],
                                               strides=2)

            # II
            features = tf.layers.conv2d(inputs=features,
                                        filters=32,
                                        strides=[2, 2],
                                        kernel_size=[5, 5],
                                        padding="same",
                                        activation=tf.nn.relu)

            features = tf.nn.relu(tf.layers.batch_normalization(features, training=self.is_training))

            features = tf.layers.max_pooling2d(inputs=features,
                                               pool_size=[2, 2],
                                               strides=2)

            features = tf.layers.flatten(features, name="flatten")

            dropout = 0.3
            features = tf.layers.dropout(features,
                                         rate=dropout,
                                         training=self.is_training,
                                         name="dropout_layer")


            output_layer = tf.layers.dense(features, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)

            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()

            start_learn_rate = 0.005
            final_learn_rate = 0.000005
            batches_per_epoch = mnist.train.num_examples // args.batch_size
            decay_rate = np.power(final_learn_rate / start_learn_rate, 1 / (args.epochs - 1))
            self.learning_rate = tf.train.exponential_decay(start_learn_rate,
                                                            global_step,
                                                            batches_per_epoch,
                                                            decay_rate, staircase=True)

            updated_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(updated_ops):
                self.training = tf.train.AdamOptimizer(self.learning_rate).minimize(loss,
                                                                                    global_step=global_step,
                                                                                    name="training")



            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.is_training: True})


    def evaluate(self, dataset, images, labels):
        return self.session.run([self.accuracy, self.summaries[dataset]],
                                {self.images: images, self.labels: labels, self.is_training: False})

    def predict(self, dataset, images):
        return self.session.run([self.predictions, self.summaries[dataset]],
                                {self.images: images, self.labels: [0] * len(images), self.is_training: False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
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
    mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42,
                                            source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        evl = network.evaluate("dev", mnist.validation.images, mnist.validation.labels)[0]

    test_labels = network.predict("test", mnist.test.images)[0]

    with open("mnist_competition_test.txt", "w") as test_file:
        for label in test_labels:
            print(label, file=test_file)

