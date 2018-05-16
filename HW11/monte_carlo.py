#!/usr/bin/env python3
import numpy as np
import cart_pole_evaluator
from collections import defaultdict


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=None, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    print("states", env.states)
    print("actions", env.actions)

    # Implement Monte-Carlo RL algorithm.

    sums = defaultdict(float)
    counts = defaultdict(float)

    Q = defaultdict(lambda: np.zeros(env.actions))
    num_episodes = 12000
    discount_factor = 1.0
    epsilon = 0.1

    # Policy function
    def policy(observation):
        A = np.ones(env.actions) * epsilon / env.actions
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    # Train + Evaluate
    for i_episode in range(1, num_episodes + 1):
        episode = []
        state_actions = []
        if num_episodes - 100 > i_episode:
            state, done = env.reset(), False
        else:
            state = env.reset(start_evaluate=True)
        for t in range(100):
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state_actions.append((state, action))
            if done:
                break
            state = next_state

        # Update Q
        for state_action in set(state_actions):
            first_state_action = next(i for i, rest in enumerate(episode)
                                       if rest[0] == state_action[0] and rest[1] == state_action[1])
            G = sum([rest[2] * (discount_factor ** i) for i, rest in enumerate(episode[first_state_action:])])
            sums[state_action] += G
            counts[state_action] += 1.0
            Q[state_action[0]][state_action[1]] = sums[state_action] / counts[state_action]

