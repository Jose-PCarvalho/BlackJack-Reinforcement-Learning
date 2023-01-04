import random

import numpy as np
from collections import defaultdict
from environment import *
from vizualization import Q_to_array , Q_to_array_double
import itertools
import sys
import plotting


def create_Q_double():
    return np.zeros(3)


def create_q_default():
    return np.zeros(2)



def mean_sqr(q1, q2):
    return np.mean(np.square(q1 - q2))


def monte_carlo(env, num_episodes, discount_factor=1.0, Q_=None):
    meanReturn = 0
    if Q_ is None:
        if env.mode == "normal":
            Q = defaultdict(create_q_default)
        else:
            Q = defaultdict(create_Q_double)
    else:
        Q = Q_
    N0 = 10
    NSA = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / np.power(NSA[state][action], 0.8)
    #epsilon = lambda state: N0 / (N0 + NS[state])
    epsilon = lambda state: 1 / np.sqrt(NS[state])
    actions = env.get_actions()

    def epsilonGreedy(state):
        eps = epsilon(state)
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            action = np.argmax(Q[state])
        return action

    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        wins=np.zeros(num_episodes))
    for episode in range(num_episodes):

        terminated = False
        SAR = list()
        state = env.reset()
        # run an episode
        t = 0
        while not terminated:
            NS[state] += 1
            action = epsilonGreedy(state)
            NSA[state][action] += 1
            state_new, terminated, r = env.step(action)
            SAR.append([state, action, r])
            state = state_new
            # Update statistics
            stats.episode_rewards[episode] += r
            stats.episode_lengths[episode] = t
            t += 1
        # Update Q
        Returns = sum([sar[2] for sar in SAR])  # sum all rewards
        for sar in SAR:
            Q[sar[0]][sar[1]] += alpha(sar[0], sar[1]) * (Returns - Q[sar[0]][sar[1]])  # Weighted mean

        meanReturn = meanReturn + 1 / (episode + 1) * (Returns - meanReturn)  # for printing only
        if r < 0:
            stats.wins[episode] += 1

        if episode % 10000 == 0:
            print("Monte Carlo - Episode %i/%i" % (episode, num_episodes))
            sys.stdout.flush()

    return Q, stats


def q_learning(env, num_episodes, discount_factor=1.0):
    if env.mode == "normal":
        Q = defaultdict(create_q_default)
    else:
        Q = defaultdict(create_Q_double)
    N0 = 10
    NSA = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / np.power(NSA[state][action], 0.8)
    #epsilon = lambda state: N0 / (N0 + NS[state])
    epsilon = lambda state: 1 / np.sqrt(NS[state])
    actions = env.get_actions()
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        wins=np.zeros(num_episodes))

    def epsilonGreedy(state):
        eps = epsilon(state)
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            # exploitation
            action = np.argmax(Q[state])
        return action

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 10000 == 0:
            print("\rQ Learning - Episode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            NS[state] += 1
            action = epsilonGreedy(state)
            NSA[state][action] += 1
            # AQUI
            # Não seria done, reward? Não, precisas de saber qual é o próximo estado
            next_state, done, reward = env.step(action)
            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha(state, action) * td_delta
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                if reward < 0:
                    stats.wins[i_episode] = 1
                break
            state = next_state

    return Q, stats


def sarsa(env, num_episodes, discount_factor=1.0):
    if env.mode == "normal":
        Q = defaultdict(create_q_default)
    else:
        Q = defaultdict(create_Q_double)
    N0 = 10
    NSA = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / np.power(NSA[state][action], 0.8)
    #epsilon = lambda state: N0 / (N0 + NS[state])
    epsilon = lambda state: 1 / np.sqrt(NS[state])
    actions = env.get_actions()
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        wins=np.zeros(num_episodes))

    def epsilonGreedy(state):
        eps = epsilon(state)
        if eps > 1:
            eps = 1
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            # exploitation
            action = np.argmax(Q[state])
        return action

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 10000 == 0:
            print("\rSARSA - Episode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        action = epsilonGreedy(state)
        for t in itertools.count():
            NS[state] += 1
            NSA[state][action] += 1
            # Mesma coisa
            next_state, done, reward = env.step(action)
            # TD Update
            # Não seria epsilongreedy(nex_state)? Sim, foi falha.
            next_action = epsilonGreedy(next_state)
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha(state, action) * td_error
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                if reward < 0:
                    stats.wins[i_episode] = 1
                break
            state = next_state
            action = next_action

    return Q, stats


def sarsa_lambda(env, num_episodes, ld=0.2, discount_factor=1,Q_baseline=None):  # ld=lambda
    if env.mode == "normal":
        Q = defaultdict(create_q_default)
    else:
        Q = defaultdict(create_Q_double)
    np.random.seed(27)
    N0 = 100
    NSA = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / NSA[state][action]
    epsilon = lambda state: N0 / (N0 + NS[state])
    actions = env.get_actions()
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        wins=np.zeros(num_episodes))

    if Q_baseline is not None:
        if env.mode=="normal":
            Q_b=Q_to_array(Q_baseline)
        else:
            Q_b = Q_to_array_double(Q_baseline)

    def epsilonGreedy(state):
        eps = max(0.01, epsilon(state))
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            # exploitation
            action = np.argmax(Q[state])
        return action

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 10000 == 0:
            print("\r SARSA LAMDA - Episode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        visited_states = []
        if env.mode == "normal":
            e_traces = defaultdict(create_q_default)
        else:
            e_traces = defaultdict(create_Q_double)
        action = epsilonGreedy(state)
        for t in itertools.count():
            NS[state] += 1
            visited_states.append(state)
            NSA[state][action] += 1
            next_state, done, reward = env.step(action)
            # TD Update
            next_action = epsilonGreedy(next_state)
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            e_traces[state][action] = (1 - alpha(state, action)) * e_traces[state][action] + 1
            #e_traces[state][action] += + 1 #acc
            for s in visited_states:
                for a in actions:
                    if NSA[s][a] > 0:
                        Q[s][a] += alpha(s, a) * td_error * e_traces[s][a]
                        e_traces[s][a] = ld * discount_factor * e_traces[s][a]
            stats.episode_rewards[i_episode] += reward

            if Q_baseline is not None:
                if env.mode == "normal":
                    stats.episode_lengths[i_episode] = mean_sqr(Q_to_array(Q), Q_b)
                else:
                    stats.episode_lengths[i_episode] = mean_sqr(Q_to_array_double(Q), Q_b)

            if done:
                if reward > 0:
                    stats.wins[i_episode] = 1
                break
            state = next_state
            action = next_action
    return Q, stats


def watkins_q(env, num_episodes, ld=0.2, discount_factor=1.0,Q_baseline=None):
    if env.mode == "normal":
        Q = defaultdict(create_q_default)
    else:
        Q = defaultdict(create_Q_double)
    N0 = 100
    np.random.seed(27)
    NSA = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / NSA[state][action]
    epsilon = lambda state: N0 / (N0 + NS[state])
    actions = env.get_actions()
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        wins=np.zeros(num_episodes))
    if Q_baseline is not None:
        if env.mode == "normal":
            Q_b = Q_to_array(Q_baseline)
        else:
            Q_b = Q_to_array_double(Q_baseline)

    def epsilonGreedy(state):
        eps = max(0.01, epsilon(state))
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            # exploitation
            action = np.argmax(Q[state])
        return action

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 10000 == 0:
            print("\r Q - LAMDA - Episode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        visited_states = []
        if env.mode == "normal":
            e_traces = defaultdict(create_q_default)
        else:
            e_traces = defaultdict(create_Q_double)
        action = epsilonGreedy(state)
        for t in itertools.count():
            NS[state] += 1
            visited_states.append(state)
            NSA[state][action] += 1
            next_state, done, reward = env.step(action)
            # TD Update
            best_flag = False
            next_action = epsilonGreedy(next_state)
            best_next_action = np.argmax(Q[next_state])
            if Q[next_state][next_action] == Q[next_state][best_next_action]:
                best_flag = True
                best_next_action = next_action

            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            e_traces[state][action] = (1 - alpha(state, action)) * e_traces[state][action] + 1
            #e_traces[state][action] += + 1 #acc
            for s in visited_states:
                for a in actions:
                    if NSA[s][a] > 0:
                        Q[s][a] += alpha(s, a) * td_error * e_traces[s][a]
                        if best_flag:
                            e_traces[s][a] = ld * discount_factor * e_traces[s][a]
                        else:
                            e_traces[s][a] = 0
            stats.episode_rewards[i_episode] += reward
            if Q_baseline is not None:
                if env.mode=="normal":
                    stats.episode_lengths[i_episode] = mean_sqr(Q_to_array(Q), Q_b)
                else:
                    stats.episode_lengths[i_episode] = mean_sqr(Q_to_array_double(Q), Q_b)
            if done:
                if reward > 0:
                    stats.wins[i_episode] = 1
                break
            state = next_state
            action = next_action
    return Q, stats


def double_q_learning(env, num_episodes, discount_factor=1.0):
    if env.mode == "normal":
        Q_A = defaultdict(create_q_default)
        Q_B = defaultdict(create_q_default)
        Q_Final = defaultdict(create_q_default)
    else:
        Q_A = defaultdict(create_Q_double)
        Q_B = defaultdict(create_Q_double)
        Q_Final = defaultdict(create_Q_double)
    N0 = 100
    A_OR_B = ["A", "B"]
    NSA_A = defaultdict(lambda: np.zeros(env.action_space_n))
    NSA_B = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha_A = lambda state, action: 1 / NSA_A[state][action]
    alpha_B = lambda state, action: 1 / NSA_B[state][action]
    epsilon = lambda state: N0 / (N0 + NS[state])
    actions = env.get_actions()
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        wins=np.zeros(num_episodes))

    def epsilonGreedy(state):
        eps = epsilon(state)
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            # exploitation
            action = np.argmax(Q_A[state] + Q_B[state])
        return action

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 10000 == 0:
            print("\rDouble Q Episode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        for t in itertools.count():
            NS[state] += 1
            action = epsilonGreedy(state)
            next_state, done, reward = env.step(action)
            Update = random.choice(A_OR_B)
            # TD Update
            if Update == "A":
                NSA_A[state][action] += 1
                best_next_action = np.argmax(Q_A[next_state])
                td_target = reward + discount_factor * Q_B[next_state][best_next_action]
                td_delta = td_target - Q_A[state][action]
                Q_A[state][action] += alpha_A(state, action) * td_delta
            if Update == "B":
                NSA_B[state][action] += 1
                best_next_action = np.argmax(Q_B[next_state])
                td_target = reward + discount_factor * Q_A[next_state][best_next_action]
                td_delta = td_target - Q_B[state][action]
                Q_B[state][action] += alpha_B(state, action) * td_delta
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break
            state = next_state
        if reward > 0:
            stats.wins[i_episode] = 1
    for s in Q_A.keys():
        for a in actions:
            Q_Final[s][a] = 0.5 * (Q_A[s][a] + Q_B[s][a])

    return Q_Final, stats
