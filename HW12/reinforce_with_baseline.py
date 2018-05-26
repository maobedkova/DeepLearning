#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

import cart_pole_evaluator

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, state_shape, num_actions):
        with self.session.graph.as_default():
            # Input states
            self.states = tf.placeholder(tf.float32, [None] + state_shape)
            # Chosen actions (used for training)
            self.actions = tf.placeholder(tf.int32, [None])
            # Observed returns (used for training)
            self.returns = tf.placeholder(tf.float32, [None])

            # Compute the action logits

            # Add a fully connected layer processing self.states, with args.hidden_layer neurons
            # and some non-linear activation.
            dense_I = tf.layers.Dense(units=args.hidden_layer, activation=tf.nn.relu)
            hidden_layer = dense_I(self.states)

            # Compute `logits` using another dense layer with
            # `num_actions` outputs (utilizing no activation function).
            dense_II = tf.layers.Dense(units=num_actions, activation=None)
            logits = dense_II(hidden_layer)

            # Compute the `self.probabilities` from the `logits`.
            self.probabilities = tf.nn.softmax(logits)

            # Compute `baseline`, by starting with a fully connected layer processing `self.states` into
            # args.hidden_layer outputs using some non-linear activation, and then employing another
            # densely connected layer with one output and no activation. Modify the result to have
            # shape `[batch_size]` (you can use for example `[:, 0]`, see the overloaded `Tensor.__getitem__` method).
            b_hidden_layer = dense_I(self.states)
            dense_III = tf.layers.Dense(units=1, activation=None)
            baseline = dense_III(b_hidden_layer)[:, 0]

            print(baseline.get_shape())

            # Training

            # Compute final `loss` as a sum of the two following losses:
            # - softmax cross entropy loss of self.actions and `logits`.
            #   Because this is REINFORCE with a baseline, you need to weight the loss of
            #   each batch element by a difference of `self.returns` and `baseline`.
            #   Also, the gradient to `baseline` should not be propagated through this loss,
            #   so you should use `tf.stop_gradient(baseline)`.
            # - mean square error of the `self.returns` and `baseline`.
            loss_I = tf.losses.sparse_softmax_cross_entropy(self.actions, logits, self.returns - baseline)
            tf.stop_gradient(baseline)
            loss_II = tf.losses.mean_squared_error(self.returns, baseline)
            loss = loss_I + loss_II

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    def predict(self, states):
        return self.session.run(self.probabilities, {self.states: states})

    def train(self, states, actions, returns):
        self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int, help="Number of episodes to train on.")
    parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    parser.add_argument("--hidden_layer", default=20, type=int, help="Size of hidden layer.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment(discrete=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, env.state_shape, env.actions)

    evaluating = False
    while True:
        # Decide if evaluation should start (one possibility is to train for args.episodes,
        # so env.episode >= args.episodes could be used).
        evaluation = env.episode >= args.episodes

        # Train for a batch of episodes
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            state = env.reset(evaluating)
            states, actions, rewards, done = [], [], [], False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # Compute action distribution using `network.predict`
                # states = np.reshape(states, (-1, 4))
                action_distr = network.predict(states)

                print(action_distr)

                # Set `action` randomly according to the generated distribution
                # (you can use np.random.choice or any other method).
                action = np.random.choice(action_distr)

                next_state, reward, done, _ = env.step(action)

                # Accumulate states, actions and rewards.
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # Compute returns from rewards (by summing them up and
            # applying discount by `args.gamma`).
            returns = np.npv(args.gamma, np.sum(rewards))

            # Extend the batch_{states,actions,returns} using the episodic
            # {states,actions,returns}.
            batch_states.append(states)
            batch_actions.append(actions)
            batch_returns.append(returns)

        # Perform network training using batch_{states,actions,returns}.
        network.train(batch_states, batch_actions, batch_returns)
