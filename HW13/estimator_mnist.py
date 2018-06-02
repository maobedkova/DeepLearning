#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def mnist_model(features, labels, mode, params):
    # Using features["images"], compute `logits` using:
    # - convolutional layer with 8 channels, kernel size 3 and ReLu activation
    conv_8 = tf.layers.Conv2D(8, 3, activation=tf.nn.relu)
    hidden = conv_8(features["images"])

    # - max pooling layer with pool size 2 and stride 2
    max_pool = tf.layers.MaxPooling2D(2, 2)
    hidden = max_pool(hidden)

    # - convolutional layer with 16 channels, kernel size 3 and ReLu activation
    conv_16 = tf.layers.Conv2D(16, 3, activation=tf.nn.relu)
    hidden = conv_16(hidden)

    # - max pooling layer with pool size 2 and stride 2
    hidden = max_pool(hidden)

    # - flattening layer
    flatten = tf.layers.Flatten()
    hidden = flatten(hidden)

    # - dense layer with 256 neurons and ReLU activation
    dense_256 = tf.layers.Dense(256, activation=tf.nn.relu)
    hidden = dense_256(hidden)

    # - dense layer with 10 neurons and no activation
    dense_10 = tf.layers.Dense(10, activation=None)
    logits = dense_10(hidden)

    predictions = tf.argmax(logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return EstimatorSpec with `mode` and `predictions` parameters
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"labels": predictions})

    # Compute loss using `tf.losses.sparse_softmax_cross_entropy`.
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    # Create `eval_metric_ops`, a dictionary with a key "accuracy", its
    # value computed using `tf.metrics.accuracy`.
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels, predictions)}

    # Get optimizer class, using `params.get("optimizer", None)`.
    opt = params.get("optimizer", None)

    # Create optimizer, using `params.get("learning_rate", None)` parameter.
    optimizer = opt(params.get("learning_rate", None))

    # Define `train_op` as `optimizer.minimize`, with `tf.train.get_global_step` as `global_step`.
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Return EstimatorSpec with `mode`, `loss`, `train_op` and `eval_metric_ops` arguments,
        # the latter being the precomputed `eval_metric_ops`.
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.EVAL:
        # Return EstimatorSpec with `mode`, `loss`, `train_op` and `eval_metric_ops`  arguments,
        # the latter being the precomputed `eval_metric_ops`.
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)


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

    # Construct the model
    model = tf.estimator.Estimator(
        model_fn=mnist_model,
        model_dir=args.logdir,
        config=tf.estimator.RunConfig(tf_random_seed=42,
                                      session_config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                                    intra_op_parallelism_threads=args.threads)),
        params={
            "optimizer": tf.train.AdamOptimizer,
            "learning_rate": 0.001,
        })

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(".", reshape=False, seed=42)

    # Train
    for i in range(args.epochs):
        # Define input_fn using `tf.estimator.inputs.numpy_input_fn`.
        # As `x`, pass `{"images": mnist.train images}`, as `y`, pass `mnist.train.labels.astype(np.int64)`,
        # use specified batch_size, one epoch. Normally we would shuffle data with queue capacity 60000,
        # but a random seed cannot be passed to this method; hence, do _not_ shuffle data.
        input_fn = tf.estimator.inputs.numpy_input_fn({"images": mnist.train.images},
                                                      mnist.train.labels.astype(np.int64),
                                                      batch_size=args.batch_size, num_epochs=1, shuffle=False)

        # Train one epoch with `model.train` using the defined input_fn.
        # Note that the `steps` argument should be either left out or set to `None` to respect
        # the `num_epochs` specified when defining `input_fn`.
        model.train(input_fn)

        # Define validation input_fn similarly, but using `mnist.validation`.
        val_input_fn = tf.estimator.inputs.numpy_input_fn({"images": mnist.validation.images},
                                                      mnist.validation.labels.astype(np.int64),
                                                      batch_size=args.batch_size, num_epochs=1, shuffle=False)

        # Evaluate the validation data, using `model.evaluate` with `name="dev"` option
        # and print its return value (which is a dictionary with accuracy, loss and global_step).
        print(model.evaluate(val_input_fn, name="dev"))

    # Define input_fn for one epoch of `mnist.test`.
    test_input_fn = tf.estimator.inputs.numpy_input_fn({"images": mnist.test.images},
                                                 mnist.test.labels.astype(np.int64),
                                                 batch_size=args.batch_size, num_epochs=1, shuffle=False)

    # Evaluate the test set using `model.evaluate` with `name="test"` option
    # and print its return value (which is a dictionary with accuracy, loss and global_step).
    print(model.evaluate(test_input_fn, name="test"))


