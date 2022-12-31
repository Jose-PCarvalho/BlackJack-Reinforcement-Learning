import numpy as np
from collections import defaultdict
from environment import *
import itertools
import sys


def create_Q_double():
    return np.zeros(3)


def create_q_default():
    return np.zeros(2)


def monte_carlo(env, num_episodes, discount_factor=1.0):
    meanReturn = 0
    wins = 0
    if env.mode == "normal":
        Q = defaultdict(create_q_default)
    else:
        Q = defaultdict(create_Q_double)
    N0 = 100
    # AQUI
    # Preciso entender melhor como funciona esse lambda
    # Procura por lambda function, é só uma função não declarada, não é muito relevante.
    NSA = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / NSA[state][action]
    epsilon = lambda state: N0 / (N0 + NS[state])
    actions = env.get_actions()

    def epsilonGreedy(state):
        eps = max(0.01, epsilon(state))
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            # exploitation
            action = np.argmax(Q[state])
        return action

    for episode in range(num_episodes):

        terminated = False
        SAR = list()
        state = env.reset()
        # run an episode
        while not terminated:
            NS[state] += 1
            action = epsilonGreedy(state)
            # AQUI
            # Não seria [state][action]? Sim, Obrigado.
            NSA[state][action] += 1
            state_new, terminated, r = env.step(action)
            SAR.append([state, action, r])
            state = state_new
        # Update Q
        Returns = sum([sar[2] for sar in SAR])  # sum all rewards
        for sar in SAR:
            Q[sar[0]][sar[1]] += alpha(sar[0], sar[1]) * (Returns - Q[sar[0]][sar[1]])  # Weighted mean

        meanReturn = meanReturn + 1 / (episode + 1) * (Returns - meanReturn)  # for printing only
        if r == 1:
            wins += 1

        if episode % 10000 == 0:
            print("Episode %i, Mean-Return %.3f, Wins %.2f" % (episode, meanReturn, wins / (episode + 1)))


def q_learning(env, num_episodes, discount_factor=1.0):
    meanReturn = 0
    wins = 0
    if env.mode == "normal":
        Q = defaultdict(create_q_default)
    else:
        Q = defaultdict(create_Q_double)
    N0 = 100
    NSA = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / NSA[state][action]
    epsilon = lambda state: N0 / (N0 + NS[state])
    actions = env.get_actions()

    def epsilonGreedy(state):
        eps = max(0.01, epsilon(state))
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            # exploitation
            action = np.argmax(Q[state])
        return action

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            action = epsilonGreedy(state)
            NS[state] += 1
            NSA[state][action] += 1
            # AQUI
            # Não seria done, reward? Não, precisas de saber qual é o próximo estado
            next_state, done, reward = env.step(action)
            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha(state, action) * td_delta

            if done:
                break
            state = next_state

    return Q


def sarsa(env, num_episodes, discount_factor=1.0):
    meanReturn = 0
    wins = 0
    if env.mode == "normal":
        Q = defaultdict(create_q_default)
    else:
        Q = defaultdict(create_Q_double)
    N0 = 100
    NSA = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / NSA[state][action]
    epsilon = lambda state: N0 / (N0 + NS[state])
    actions = env.get_actions()

    def epsilonGreedy(state):
        eps = max(0.01, epsilon(state))
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            # exploitation
            action = np.argmax(Q[state])
        return action

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 10000 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            print("\n Wins %.2f" % (wins / 10000))
            wins = 0
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

            if done:
                if reward == 1:
                    wins += 1
                break
            state = next_state
            action = next_action

    return Q


def sarsa_lambda(env, num_episodes, ld=1, discount_factor=1.0):  # ld=lambda
    meanReturn = 0
    wins = 0
    if env.mode == "normal":
        Q = defaultdict(create_q_default)
    else:
        Q = defaultdict(create_Q_double)
    N0 = 100
    NSA = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / NSA[state][action]
    epsilon = lambda state: N0 / (N0 + NS[state])
    actions = env.get_actions()

    def epsilonGreedy(state):
        eps = max(0.01, epsilon(state))
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            # exploitation
            action = np.argmax(Q[state])
        return action

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            print("\n Wins %.2f" %( wins/(i_episode+1)))
            sys.stdout.flush()
        state = env.reset()
        if env.mode == "normal":
            e_traces = defaultdict(create_q_default)
        else:
            e_traces = defaultdict(create_Q_double)
        action = epsilonGreedy(state)
        for t in itertools.count():
            NS[state] += 1
            NSA[state][action] += 1
            next_state, done, reward = env.step(action)
            # TD Update
            next_action = epsilonGreedy(next_state)
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            e_traces[state][action] = (1 - alpha(state, action)) * e_traces[state][action] + 1
            for s in Q.keys():
                for a in actions:
                    # AQUI, não é preciso o if porque se o e_trace=0 o Q continua igual
                    # Mas havia um bug na formula, obrigado.
                    Q[s][a] += alpha(s, a) * td_error * e_traces[s][a]
                    e_traces[s][a] = ld * discount_factor * e_traces[s][a]
            if done:
                if reward == 1:
                    wins += 1
                break
            state = next_state
            action = next_action

    return Q


def watkins_q(env, num_episodes, ld=1.0, discount_factor=1.0):
    meanReturn = 0
    wins = 0
    if env.mode == "normal":
        Q = defaultdict(create_q_default)
    else:
        Q = defaultdict(create_Q_double)
    N0 = 100
    NSA = defaultdict(lambda: np.zeros(env.action_space_n))
    NS = defaultdict(lambda: np.zeros(1))
    alpha = lambda state, action: 1 / NSA[state][action]
    epsilon = lambda state: N0 / (N0 + NS[state])
    actions = env.get_actions()

    def epsilonGreedy(state):
        eps = max(0.01, epsilon(state))
        if np.random.random() < eps:
            action = np.random.choice(actions)
        else:
            # exploitation
            action = np.argmax(Q[state])
        return action

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        if env.mode == "normal":
            e_traces = defaultdict(create_q_default)
        else:
            e_traces = defaultdict(create_Q_double())

        action = epsilonGreedy(state)
        for t in itertools.count():
            best_flag = False
            NS[state] += 1
            NSA[state][action] += 1
            next_state, reward, done = env.step(action)
            # TD Update
            next_action = epsilonGreedy(next_state)
            best_next_action = np.argmax(Q[next_state])
            if Q[next_state][next_action] == Q[next_state][best_next_action]:
                best_next_action = next_action
                best_flag = True
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            e_traces[state][action] = (1 - alpha(state, action)) * e_traces[state][action] + 1
            for s in Q.keys():
                for a in actions:
                    Q[s][a] += alpha(s, a) * td_error * e_traces[s][a]
                    if best_flag:
                        e_traces[s][a] = ld * discount_factor * e_traces[s][a]
                    else:
                        e_traces[s][a] = 0

            if done:
                break
            state = next_state
            action = next_action
    return Q
