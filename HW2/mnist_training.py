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
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

            # Computation
            flattened_images = tf.layers.flatten(self.images, name="flatten")
            hidden_layer = tf.layers.dense(flattened_images, args.hidden_layer, activation=tf.nn.relu, name="hidden_layer")
            output_layer = tf.layers.dense(hidden_layer, self.LABELS, activation=None, name="output_layer")
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()

            self.learning_rate = args.learning_rate

            if args.learning_rate_final:
                decay_rate = np.power(args.learning_rate_final / args.learning_rate, 1 / (args.epochs - 1))
                self.learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, batches_per_epoch,
                                                                decay_rate, staircase=True)

            if args.optimizer == "SGD" and args.momentum:
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=args.momentum, name="Momentum")
            elif args.optimizer == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name="GradientDescent")
            elif args.optimizer == "Adam":
                optimizer = tf.train.AdamOptimizer(self.learning_rate, name="Adam")

            self.training = optimizer.minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy),
                                           tf.contrib.summary.scalar("train/learning_rate", self.learning_rate)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels})

    def evaluate(self, dataset, images, labels):
        return self.session.run([self.accuracy, self.summaries[dataset]], {self.images: images, self.labels: labels})


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
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
    parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
    parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")
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
    batches_per_epoch = mnist.train.num_examples // args.batch_size

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        for b in range(batches_per_epoch):
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
    accuracy = network.evaluate("test", mnist.test.images, mnist.test.labels)[0]

    print("{:.2f}".format(100 * accuracy))
