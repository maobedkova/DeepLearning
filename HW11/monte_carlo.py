#!/usr/bin/env python3
import numpy as np
import cart_pole_evaluator


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.005, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = cart_pole_evaluator.environment()

    # Implement Monte-Carlo RL algorithm.

    policy = np.full((env.states, env.actions), 1 / env.actions)
    Q = np.zeros((env.states, env.actions), dtype=float)
    A = np.zeros((env.states, env.actions), dtype=int)

    # Training
    training = True
    rewards = []
    while training:
        # Generate episode
        episode = []
        sum_reward = 0
        state, done = env.reset(), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()
            action = np.random.choice(env.actions, replace=False, p=policy[state, :])
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            sum_reward += reward

        # Condition for finishing the training
        rewards.append(sum_reward)
        epsilon = args.epsilon - (np.mean(rewards[-400:]) / 500.0) * (args.epsilon - args.epsilon_final)
        # epsilon = args.epsilon

        # print(np.mean(rewards[-400:]))
        if np.mean(rewards[-400:]) > 475.0:
            training = False

        # Update policy
        G = 0
        for (state, action, reward) in reversed(episode):
            G += reward
            avg = A[state, action]
            A[state, action] += 1
            Q[state, action] = ((Q[state, action] * avg) + G) / (avg + 1)
            best = np.argmax(Q[state, :])
            for action_id in range(env.actions):
                if action_id == best:
                    policy[state, action_id] = 1 - epsilon + (epsilon / env.actions)
                else:
                    policy[state, action_id] = epsilon / env.actions

    # Perform last 100 evaluation episodes
    for i in range(100):
        state, done = env.reset(start_evaluate=True), False
        while not done:
            action = np.random.choice(env.actions, replace=False, p=policy[state, :])
            next_state, reward, done, _ = env.step(action)
            state = next_state