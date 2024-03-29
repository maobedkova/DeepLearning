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
            for kernel_size in range(2, 16):
                hidden_layer = tf.layers.conv1d(embedded_word_ids, filters=24,
                                                kernel_size=kernel_size, strides=1, padding=\"valid\")
              #  hidden_layer = tf.layers.dropout(hidden_layer, rate=0.3, training=self.is_training, name=\"dropout\")
                feats.append(tf.layers.max_pooling1d(hidden_layer, 500, strides=1, padding=\"same\")[:, 1, :])

            concat_feats = tf.concat(feats, axis=-1)

            output_layer = tf.layers.dense(concat_feats, units=num_languages, activation=None)
            # output_layer = tf.layers.dropout(output_layer, rate=0.3, training=self.is_training, name=\"dropout\")

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

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;1Mqa3S9st6b0I$g{*AAujET%e6x(40Q*l#upU(zDi08t#TfqFm=ruK2-MT}&zrU$uzNkHilaE9P&l#^azG&Ii~k=5L~+!{O_2@Cz4&H<;p-10H$%*kzMVcQFhJ5hr_N(mm1omE9n!B2F^k&qZSNejh89q$559VFFe;(X*bDgPq=b^JEPF8OmaTnj`L`CydpyO{t%h!LFP4wv!~l^GgXC|{fg)e@CffXg7~tsrs<U%BB(}aQUxr4k1`{s$eZrRnb}tQ>Dw}L=BKg;65r@U}YvqgOWuIRzKQ$@*2A`A4)I}G}KvDux28LkL(X26CJyUlb2VZN}4{MvrZII6`Zf4ZJaG$)IpUo5gOkeX-h9ItxLQ@5nQPEY{O^~(yGD-P^iv4yqEfSAUXIeR9xRkm4r<|K6ebAdA^DH)!9>|_AuaA{LP!@;#I0yv)KrK=WM>yM#y}Fd{EX1Smh()vQbv;xy*IE?20qAg?gjv?M{NDp64XhE)nzoSIm=#N)>0`~A9Z<j<99fRY%Luig`4kM8a>11i_i-nF<atvM9~7C)u9aUK8#T(~P+e^rmDvH@eM;vMKgP7*lkM?AwdyR;Ux!)79X4h<EcUjDiAu}{Q%2~2J(YwWK#}F!Ui7voA>ULD<=ds=pStNwhtbr#ln7dvAp+cUn7bI|kYJ)i50P&|IYXv#nWZnc^@B=fxFELIO<rE8H_L4&a(gBQknsA+KFo!OFv`FYCvdHm7=OxkXbmn^!ihzCZ+_fPiwg5&#OhqS)$+8;EGOd9h=Uugjro7+En6m8X<yShmh`2#fI?>4y|MAVfKl1DSuBvse6M}y-=sa+u4dIN`B;D4X!~zQL(u@aR+0U+H~5ns6@X}8IgX1ubNKQdRF5<(L)I+JrG=p*VD0QJdJHaNR@hptC*UWBpp(^WZKMP3U3_t5z~gnH(Cm&H-0;>qZ+JgMKj`hQ7aLgBByqW2(rzaK&D&~YQ!}1an1hqWJq%uRS&LHZ8~^|SAi4PMhzUy^00E{3up$5enh)!xvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
