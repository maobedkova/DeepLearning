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

            word_embeddings = tf.get_variable(\"word_embeddings\", [num_words, 128])
            embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, self.word_ids)

            feats = []
            for kernel_size in range(2, 11):
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
    parser.add_argument(\"--epochs\", default=20, type=int, help=\"Number of epochs.\")
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

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;1Mqa30(js6b0I%KSMIrc?^VQfgIh^y!Mb=f%?=cFo{D=e{@<TlTqrT7fppXUU<Vdan@luTAWNtNX3*dPNZulb7KzFZ`;PjvO|BR*~Bsg0Lp!@ekHzMs5|-h<IV~WJp7wg-kE1}t~%*taDtcHp=1Rho)CP39tkQO73M2oRPj2T6Sx%J`-GSiE%Egv0s4@u8{`d{Q<T<D<~W-#INutjv8S<j)ulRx#!u(t8!(!_I{)?eakKjd6|1Jg595eC5>JwL+Zn=;-rs0|Pw&(MAZNjjiOIAJ<bAn@lk@IrXKY~Hcyo#o`Z@*An}zF)wk@;_qL1otBC3vtP6123MTIRU9^WFRZYYF9#A}*Pae{6iz5Iqk;M^s1t*!Ozq3_G+D-2ZJsP($$H**jn)?Fzc6?ChjO42lPH32_D>gZ54$A<8IR9Xx%dEu^rr}D9>Mf4nRie&8MysrIcP54*=263O;CllP%32aIl(R#n->)l7Iiv?CG7HB=tV>(k?tgj<uv^hQdfu3wqd0rjaf0yNh`lFhJ9+HU>V^p*@k`~`H5F{)A77r@7oPdvwG=&yF6Z-^lLrZJ5M6PPmp&AVwT#o^UO&+%Nef_(vkYq?!>|T)t)VxEPX(^9c78YqjX}Yo}G*fHEt}#UMZM#iwj>d)4ky)J&@pZCry65wjJb`xiEf40NG31nZtVv4?jbrdSMlp9$^2}i|oP_;Ntnl{;uSpg{G?&MUj3fMS-`Vg`s2g<dIW4;J>3z{#r0(N?kUk_8a9lD@AuiAp8!mmq(kMW6vu3*B{}n;5CP~#qM{_#IqUFYf5RGv>656M7*V4kpMJ~Xv1Mq(u?T?>-W71s7ciM(qWG|6J32F&G-vqA!eRF4~Z?vY>&7^^MT4|j4M9$P#JERUxm#R74je81DH7UD1eZUSyNT9Xm;hVMKE;ABLw0J7d3d&N#?FAx{VrXETbROT8a|NF`dbndGhuFTo_c{V{>%x1A9@W>o`EXs-FaIZ=00000T|X!ZEm~O!00E^2up$5eG<|;~vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
