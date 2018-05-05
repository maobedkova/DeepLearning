#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import morpho_dataset

class MorphoAnalyzer:
    """ Loader for data of morphological analyzer.

    The loaded analyzer provides an only method `get(word)` returning
    a list of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    class LemmaTag:
        def __init__(self, lemma, tag):
            self.lemma = lemma
            self.tag = tag

    def __init__(self, filename):
        self.analyses = {}

        with open(filename, "r", encoding="utf-8") as analyzer_file:
            for line in analyzer_file:
                line = line.rstrip("\n")
                columns = line.split("\t")

                analyses = []
                for i in range(1, len(columns) - 1, 2):
                    analyses.append(MorphoAnalyzer.LemmaTag(columns[i], columns[i + 1]))
                self.analyses[columns[0]] = analyses

    def get(self, word):
        return self.analyses.get(word, [])


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, source_chars, target_chars, bow, eow):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.source_ids = tf.placeholder(tf.int32, [None, None], name="source_ids")
            self.source_seqs = tf.placeholder(tf.int32, [None, None], name="source_seqs")
            self.source_seq_lens = tf.placeholder(tf.int32, [None], name="source_seq_lens")
            self.target_ids = tf.placeholder(tf.int32, [None, None], name="target_ids")
            self.target_seqs = tf.placeholder(tf.int32, [None, None], name="target_seqs")
            self.target_seq_lens = tf.placeholder(tf.int32, [None], name="target_seq_lens")

            # Training. The rest of the code assumes that
            # - when training the decoder, the output layer with logis for each generated
            #   character is in `output_layer` and the corresponding predictions are in
            #   `self.predictions_training`.
            # - the `target_ids` contains the gold generated characters
            # - the `target_lens` contains number of valid characters for each lemma
            # - when running decoder inference, the predictions are in `self.predictions`
            #   and their lengths in `self.prediction_lens`.

            # Append EOW after target_seqs
            target_ids = tf.reverse_sequence(self.target_seqs, self.target_seq_lens, 1)
            target_ids = tf.pad(target_ids, [[0, 0], [1, 0]], constant_values=eow)
            target_seq_lens = self.target_seq_lens + 1
            target_ids = tf.reverse_sequence(target_ids, target_seq_lens, 1)

            # Encoder
            # Generate source embeddings for source chars, of shape [source_chars, args.char_dim].
            source_embeddings = tf.get_variable("source_embeddings", [source_chars, args.char_dim])

            # Embed the self.source_seqs using the source embeddings.
            embedded_source_seqs = tf.nn.embedding_lookup(source_embeddings, self.source_seqs)

            # Using a GRU with dimension args.rnn_dim, process the embedded self.source_seqs
            # using bidirectional RNN. Store the summed fwd and bwd outputs in `source_encoded`
            # and the summed fwd and bwd states into `source_states`.
            rnn_cell_fwd = tf.nn.rnn_cell.GRUCell(num_units=args.rnn_dim)
            rnn_cell_bwd = tf.nn.rnn_cell.GRUCell(num_units=args.rnn_dim)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fwd, rnn_cell_bwd, embedded_source_seqs,
                                                              sequence_length=self.source_seq_lens, dtype=tf.float32)
            source_encoded = tf.add(outputs[0], outputs[1])
            source_states = tf.add(states[0], states[1])

            # Index the unique words using self.source_ids and self.target_ids.
            sentence_mask = tf.sequence_mask(self.sentence_lens)
            source_encoded = tf.boolean_mask(tf.nn.embedding_lookup(source_encoded, self.source_ids), sentence_mask)
            source_states = tf.boolean_mask(tf.nn.embedding_lookup(source_states, self.source_ids), sentence_mask)
            source_lens = tf.boolean_mask(tf.nn.embedding_lookup(self.source_seq_lens, self.source_ids), sentence_mask)

            target_ids = tf.boolean_mask(tf.nn.embedding_lookup(target_ids, self.target_ids), sentence_mask)
            target_lens = tf.boolean_mask(tf.nn.embedding_lookup(target_seq_lens, self.target_ids), sentence_mask)

            # Decoder
            # Generate target embeddings for target chars, of shape [target_chars, args.char_dim].
            target_embeddings = tf.get_variable("target_embeddings", [target_chars, args.char_dim])

            # Embed the target_seqs using the target embeddings.
            embedded_target_seqs = tf.nn.embedding_lookup(target_embeddings, target_ids)

            # Generate a decoder GRU with dimension args.rnn_dim.
            rnn_decoder = tf.nn.rnn_cell.GRUCell(num_units=args.rnn_dim)

            # Create a `decoder_layer` -- a fully connected layer with
            # target_chars neurons used in the decoder to classify into target characters.
            decoder_layer = tf.layers.Dense(units=target_chars, activation=None)

            # Attention
            # Generate three fully connected layers without activations:
            # - `source_layer` with args.rnn_dim units
            # - `state_layer` with args.rnn_dim units
            # - `weight_layer` with 1 unit
            source_layer = tf.layers.Dense(args.rnn_dim, activation=None)
            state_layer = tf.layers.Dense(args.rnn_dim, activation=None)
            weight_layer = tf.layers.Dense(1, activation=None)

            def with_attention(inputs, states):
                # Generate the attention

                # Project source_encoded using source_layer.
                source_projection = source_layer(source_encoded)

                # Change shape of states from [a, b] to [a, 1, b] and project it using state_layer.
                states = tf.expand_dims(states, 1)
                state_projection = state_layer(states)

                # Sum the two above projections, apply tf.tanh and project the result using weight_layer.
                # The result has shape [x, y, 1].
                weight_projection = weight_layer(tf.tanh(tf.add(source_projection, state_projection)))

                # Apply tf.nn.softmax to the latest result, using axis corresponding to source characters.
                softmax_layer = tf.nn.softmax(weight_projection, axis=1)

                # Multiply the source_encoded by the latest result, and sum the results with respect
                # to the axis corresponding to source characters. This is the final attention.
                tmp = tf.multiply(source_encoded, softmax_layer)
                attention = tf.reduce_sum(tmp, axis=1)

                # Return concatenation of inputs and the computed attention.
                return tf.concat([inputs, attention], 1)

            # The DecoderTraining will be used during training. It will output logits for each
            # target character.
            class DecoderTraining(tf.contrib.seq2seq.Decoder):
                @property
                def batch_size(self): return tf.shape(source_states)[
                    0]  # Return size of the batch, using for example source_states size

                @property
                def output_dtype(self): return tf.float32  # Type for logits of target characters

                @property
                def output_size(self): return target_chars  # Length of logits for every output

                def initialize(self, name=None):
                    finished = target_lens <= 0  # False if target_lens > 0, True otherwise
                    states = source_states  # Initial decoder state to use
                    inputs = with_attention(tf.nn.embedding_lookup(target_embeddings, tf.fill([self.batch_size], bow)),
                                            states)
                    # Call with_attention on the embedded BOW characters of shape [self.batch_size].
                    # You can use tf.fill to generate BOWs of appropriate size.
                    return finished, inputs, states

                def step(self, time, inputs, states, name=None):
                    outputs, states = rnn_decoder(inputs, states)  # Run the decoder GRU cell using inputs and states.
                    outputs = decoder_layer(outputs)  # Apply the decoder_layer on outputs.
                    next_input = with_attention(embedded_target_seqs[:, time],
                                                states)  # Next input is with_attention called on words with index `time` in target_embedded.
                    finished = tf.less_equal(target_lens, time + 1)  # False if target_lens > time + 1, True otherwise.
                    return outputs, states, next_input, finished

            output_layer, _, _ = tf.contrib.seq2seq.dynamic_decode(DecoderTraining())
            self.predictions_training = tf.argmax(output_layer, axis=2, output_type=tf.int32)

            # The DecoderPrediction will be used during prediction. It will
            # directly output the predicted target characters.
            class DecoderPrediction(tf.contrib.seq2seq.Decoder):
                @property
                def batch_size(self): return tf.shape(source_states)[
                    0]  # Return size of the batch, using for example source_states size

                @property
                def output_dtype(self): return tf.int32  # Type for predicted target characters

                @property
                def output_size(self): return 1  # Will return just one output

                def initialize(self, name=None):
                    finished = tf.fill([self.batch_size], False)  # False of shape [self.batch_size].
                    states = source_states  # Initial decoder state to use.
                    inputs = with_attention(tf.nn.embedding_lookup(target_embeddings, tf.fill([self.batch_size], bow)),
                                            states)
                    # Call with_attention on the embedded BOW characters of shape [self.batch_size].
                    # You can use tf.fill to generate BOWs of appropriate size.
                    return finished, inputs, states

                def step(self, time, inputs, states, name=None):
                    outputs, states = rnn_decoder(inputs, states)  # Run the decoder GRU cell using inputs and states.
                    outputs = decoder_layer(outputs)  # Apply the decoder_layer on outputs.
                    # Use tf.argmax to choose most probable class (supply parameter `output_type=tf.int32`).
                    outputs = tf.argmax(outputs, axis=1, output_type=tf.int32)
                    next_input = with_attention(tf.nn.embedding_lookup(target_embeddings, outputs),
                                                states)  # Embed `outputs` using target_embeddings and pass it to with_attention.
                    finished = tf.equal(outputs, eow)  # True where outputs==eow, False otherwise
                    # Use tf.equal for the comparison, Python's '==' is not overloaded
                    return outputs, states, next_input, finished

            self.predictions, _, self.prediction_lens = tf.contrib.seq2seq.dynamic_decode(
                DecoderPrediction(), maximum_iterations=tf.reduce_max(source_lens) + 10)


            # - the `target_ids` contains the gold generated characters
            # - the `target_lens` contains number of valid characters for each lemma


            # Training
            weights = tf.sequence_mask(target_lens, dtype=tf.float32)
            loss = tf.losses.sparse_softmax_cross_entropy(target_ids, output_layer, weights=weights)
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            accuracy_training = tf.reduce_all(tf.logical_or(
                tf.equal(self.predictions_training, target_ids),
                tf.logical_not(tf.sequence_mask(target_lens))), axis=1)
            self.current_accuracy_training, self.update_accuracy_training = tf.metrics.mean(accuracy_training)

            minimum_length = tf.minimum(tf.shape(self.predictions)[1], tf.shape(target_ids)[1])
            accuracy = tf.logical_and(
                tf.equal(self.prediction_lens, target_lens),
                tf.reduce_all(tf.logical_or(
                    tf.equal(self.predictions[:, :minimum_length], target_ids[:, :minimum_length]),
                    tf.logical_not(tf.sequence_mask(target_lens, maxlen=minimum_length))), axis=1))
            self.current_accuracy, self.update_accuracy = tf.metrics.mean(accuracy)

            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy_training)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        while not train.epoch_finished():
            sentence_lens, _, charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size, including_charseqs=True)
            self.session.run(self.reset_metrics)
            self.session.run(
                [self.training, self.summaries["train"]],
                {self.sentence_lens: sentence_lens,
                 self.source_ids: charseq_ids[train.FORMS], self.target_ids: charseq_ids[train.LEMMAS],
                 self.source_seqs: charseqs[train.FORMS], self.target_seqs: charseqs[train.LEMMAS],
                 self.source_seq_lens: charseq_lens[train.FORMS], self.target_seq_lens: charseq_lens[train.LEMMAS]})

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, _, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.source_ids: charseq_ids[train.FORMS], self.target_ids: charseq_ids[train.LEMMAS],
                              self.source_seqs: charseqs[train.FORMS], self.target_seqs: charseqs[train.LEMMAS],
                              self.source_seq_lens: charseq_lens[train.FORMS], self.target_seq_lens: charseq_lens[train.LEMMAS]})
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        lemmas = []
        while not dataset.epoch_finished():
            sentence_lens, _, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            predictions, prediction_lengths = self.session.run(
                [self.predictions, self.prediction_lens],
                {self.sentence_lens: sentence_lens, self.source_ids: charseq_ids[train.FORMS],
                 self.source_seqs: charseqs[train.FORMS], self.source_seq_lens: charseq_lens[train.FORMS]})

            for length in sentence_lens:
                lemmas.append([])
                for i in range(length):
                    lemmas[-1].append("")
                    for j in range(prediction_lengths[i] - 1):
                        lemmas[-1][-1] += train.factors[train.LEMMAS].alphabet[predictions[i][j]]
                predictions, prediction_lengths = predictions[length:], prediction_lengths[length:]

        return lemmas


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=60, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
    parser.add_argument("--char_dim", default=128, type=int, help="Character embedding dimension.")
    parser.add_argument("--rnn_dim", default=128, type=int, help="Dimension of the encoder and the decoder.")
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
    train = morpho_dataset.MorphoDataset("czech-pdt-train.txt")
    dev = morpho_dataset.MorphoDataset("czech-pdt-dev.txt", train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset("czech-pdt-test.txt", train=train, shuffle_batches=False)

    # analyzer_dictionary = MorphoAnalyzer("czech-pdt-analysis-dictionary.txt")
    # analyzer_guesser = MorphoAnalyzer("czech-pdt-analysis-guesser.txt")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, len(train.factors[train.FORMS].alphabet), len(train.factors[train.LEMMAS].alphabet),
                      train.factors[train.LEMMAS].alphabet_map["<bow>"], train.factors[train.LEMMAS].alphabet_map["<eow>"])

    # Train
    for i in range(args.epochs):
        network.train_epoch(train, args.batch_size)

        network.evaluate("dev", dev, args.batch_size)

    # Predict test data
    with open("{}/lemmatizer_sota_test.txt".format(args.logdir), "w", encoding="utf-8") as test_file:
        forms = test.factors[test.FORMS].strings
        lemmas = network.predict(test, args.batch_size)
        for s in range(len(forms)):
            for i in range(len(forms[s])):
                print("{}\t{}\t_".format(forms[s][i], lemmas[s][i]), file=test_file)
            print("", file=test_file)
