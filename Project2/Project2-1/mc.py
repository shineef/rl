#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
"""
import sys

import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hit otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    # print(observation)
    player_score, _, _ = observation
    if player_score >= 20:
        action = 0
    else:
        action = 1
    return action


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    for _ in range(n_episodes):
        episode = []
        state = env.reset()[0]
        done = False
        # print(state)

        while not done:
            action = policy(state)
            # print(env.step(action))
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G = 0  # Initialize the return (cumulative reward) to 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            # print(state)
            G = gamma * G + reward
            if state not in [s for s, _, _ in episode[:t]]:
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]

    return V


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: 
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, nA - 1)  # Choose a random action
        # print("1",action)
    else:
        action = np.argmax(Q[state])
        # print("2", Q, Q[state])
        # 2 defaultdict(<function test_epsilon_greedy.<locals>.<lambda> at 0x0000021118B789A0>, {(14, 7, True): array([0., 0., 0., 0.])}) [0. 0. 0. 0.]
    return action


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-0.1/n_episode during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # print('3', Q)
    # 3 defaultdict(<function mc_control_epsilon_greedy.<locals>.<lambda> at 0x00000246C9130AE0>, {})

    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    for episode in range(1, n_episodes + 1):
        if episode % 1000 == 0:
            # print("\rEpisode {}/{}".format(episode, n_episodes), end="")
            sys.stdout.flush()

        epsilon = max(epsilon - 0.1 / episode, 0.1)

        state = env.reset()[0]
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []

        while not done:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state

        G = 0
        for t in range(len(episode_states) - 1, -1, -1):
            state_t = episode_states[t]
            action_t = episode_actions[t]
            reward_t = episode_rewards[t]
            G = gamma * G + reward_t

            if (state_t, action_t) not in zip(episode_states[:t], episode_actions[:t]):
                returns_sum[(state_t, action_t)] += G
                returns_count[(state_t, action_t)] += 1
                Q[state_t][action_t] = returns_sum[(state_t, action_t)] / returns_count[(state_t, action_t)]

    return Q
