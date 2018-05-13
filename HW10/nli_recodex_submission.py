# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import nli_dataset

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, num_words, num_chars, num_languages):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name=\"sentence_lens\")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name=\"word_ids\")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name=\"charseqs\")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name=\"charseq_lens\")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name=\"charseq_ids\")
            self.languages = tf.placeholder(tf.int32, [None], name=\"languages\")

            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            # Training.
            # Define:
            # - loss in `loss`
            # - training in `self.training`
            # - predictions in `self.predictions`

            word_embeddings = tf.get_variable(\"word_embeddings\", [num_words, 64])
            embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, self.word_ids)

            feats = []
            for kernel_size in range(2, 6):
                hidden_layer = tf.layers.conv1d(embedded_word_ids, filters=24,
                                                kernel_size=kernel_size, strides=1, padding=\"valid\")
                #hidden_layer = tf.layers.dropout(hidden_layer, rate=0.3, training=self.is_training, name=\"dropout\")
                feats.append(tf.layers.max_pooling1d(hidden_layer, 500, strides=1, padding=\"same\")[:, 1, :])

            concat_feats = tf.concat(feats, axis=-1)

            output_layer = tf.layers.dense(concat_feats, units=num_languages, activation=None)
            #output_layer = tf.layers.dropout(output_layer, rate=0.3, training=self.is_training, name=\"dropout\")

            self.predictions = tf.argmax(output_layer, axis=-1)

            loss = tf.losses.sparse_softmax_cross_entropy(self.languages, output_layer)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name=\"training\")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.languages, self.predictions)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.size(self.sentence_lens))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", self.update_loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        while not train.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \\
                train.next_batch(batch_size)
            self.session.run(self.reset_metrics)
            self.session.run([self.training, self.summaries[\"train\"]],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                              self.languages: languages,
                              self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, languages = \\
                dataset.next_batch(batch_size)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                              self.word_ids: word_ids, self.charseq_ids: charseq_ids,
                              self.languages: languages,
                              self.is_training: False})

        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        languages = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, tags, levels, prompts, _ = \\
                dataset.next_batch(batch_size)
            languages.extend(self.session.run(self.predictions,
                                              {self.sentence_lens: sentence_lens,
                                               self.charseqs: charseqs, self.charseq_lens: charseq_lens,
                                               self.word_ids: word_ids, self.charseq_ids: charseq_ids,
					       self.is_training: False}))

        return languages


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=24, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=30, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=4, type=int, help=\"Maximum number of threads to use.\")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    train = nli_dataset.NLIDataset(\"nli-train.txt\")
    dev = nli_dataset.NLIDataset(\"nli-dev.txt\", train=train, shuffle_batches=False)
    test = nli_dataset.NLIDataset(\"nli-test.txt\", train=train, shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.vocabulary(\"words\")), len(train.vocabulary(\"chars\")), len(train.vocabulary(\"languages\")))

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        acc = network.evaluate(\"dev\", dev, args.batch_size)
        print(i, \"{:.2f}\".format(100 * acc))

    # Predict test data
    with open(\"{}/nli_test.txt\".format(args.logdir), \"w\", encoding=\"utf-8\") as test_file:
        languages = network.predict(test, args.batch_size)
        for language in languages:
            print(test.vocabulary(\"languages\")[language], file=test_file)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;1Mqa30(js6b0I$=s`D;+d<JLkLiWPQQLLoU)g8$$)#XY8g-7L%r{YuG%DP{SM}Zc-}cqh^lxr;_<~LyUQ^VaMj^$xgiIv1vKc&s<QNFldMLXG_(?~n7UYxBQ<5bC07P2WHlSxrFa>sv=PM$txic@Vy~ZJTx_Libl$P2eko9iXQ_??o(*KBdNn8Sngzl2dh3e2TtM=0{FRVkK-l(wji|m3+v*hf^lvSeN&cSi98^x;1!QIs&4f2Vw8GVf3c7pJXjK2Tj0{W6sD8JoO_AJ1N!P7_UPYg@fls8>qY#~<v*eDP?QHbc+{Ro1B3=>nYXBld^fY<psw9B>gFRK|#@&izct_-Jhy&<X8BLuF@dD2Ft0I{KAU=%8&b03QK5y47|1mJxRLbIh8E~M3{-6wZ-P<wM~#g*atXjm*+_(y&lONu~xZ@%c}e}*LFSm1tZHWYJuMZ{!X0=$IB=0O($+@%EE<6jcc#7&SOdMC-Nc%)YcNpP>R8CJ2tyETZ)flr#D*;tz(&Y?(H;7Mcecx3T<ZN~-@hFB>H1kVbtT>(atTQ+eRm%G^~n*PRZB)H<l7usLDz{LmX?rvG|2)(`FSLIl&981w(ZvCQc*j(D7K}Bxt$;VV9Ue6?&eJTpu=od<%%V%V)3L9|v(-q|KSb$9{isX0VnucalksJjx!Tr_0^nQB%kK>v7DLd>24Q(2HAgF2DSHDD_UWqaZ6=rj*Mk>_mGAsUwo)h2WCuiYswGAS#NSIrvUgp@8Yd$g8)|x+mQWHBwLMuy2nmGu#EU!`t`qr3A+uLUv#}fvHR#u2U|2Qm)wg=6mBOQRfB!LmWtC0wqq)IZrZygSt7HE~^cw7^S)eh5_j^O`<(T(eSLbiy-vRDzb;~cQoUX0$WaMQv=1eIqWlED*jo)rCBhhfM{Uhggq{3*J+@wv1t2NdGJi=E5QoUkmDs#YUL3cwp`+a*x{g)kp7qjq6Kw8-#IbxL)|8=T^7fxv6Fuz<QQ00000GkeOAxO=;|00E^2up$5eG<|;~vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())