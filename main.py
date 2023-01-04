from vizualization import viz_sequence
from environment import *
from environment import *
from algorithms import *
from plotting import *
import pickle
import dill
from collections import namedtuple

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "wins"])

env=BlackJack(mode="Double")

# Q,stats=monte_carlo(env,num_episodes=int(1e7))
# with open('results/MSE/MC_Double_Baseline.pkl', 'wb') as file:
#     dill.dump(Q, file, pickle.HIGHEST_PROTOCOL)
# with open('results/monte_carlo_stats.pkl', 'wb') as file:
#     dill.dump(stats, file, pickle.HIGHEST_PROTOCOL)
Q,stats=double_q_learning(env,num_episodes=int(1e7))
with open('results/MSE/double_q_baseline.pkl', 'wb') as file:
    dill.dump(Q, file, pickle.HIGHEST_PROTOCOL)
with open('results/double_q_learning_stats.pkl', 'wb') as file:
    dill.dump(stats, file, pickle.HIGHEST_PROTOCOL)

# Q,stats=q_learning(env,num_episodes=int(1e7))
# with open('results/q_learning.pkl', 'wb') as file:
#     dill.dump(Q, file, pickle.HIGHEST_PROTOCOL)
# with open('results/q_learning_stats.pkl', 'wb') as file:
#     dill.dump(stats, file, pickle.HIGHEST_PROTOCOL)
#
#
#
# Q,stats=sarsa(env,num_episodes=int(1e7))
# with open('results/sarsa.pkl', 'wb') as file:
#     dill.dump(Q, file, pickle.HIGHEST_PROTOCOL)
# with open('results/sarsa_stats.pkl', 'wb') as file:
#     dill.dump(stats, file, pickle.HIGHEST_PROTOCOL)
#
# # Q,stats=sarsa_lambda(env,num_episodes=int(1e6))
# # with open('results/sarsa_lambda.pkl', 'wb') as file:
# #     dill.dump(Q, file, pickle.HIGHEST_PROTOCOL)
# # with open('results/sarsa_lambda_stats.pkl', 'wb') as file:
# #     dill.dump(stats, file, pickle.HIGHEST_PROTOCOL)
#
#
# Q,stats=watkins_q(env,num_episodes=int(1e6))
# with open('results/Q_lambda.pkl', 'wb') as file:
#     dill.dump(Q, file, pickle.HIGHEST_PROTOCOL)
# with open('results/Q_lambda_stats.pkl', 'wb') as file:
#     dill.dump(stats, file, pickle.HIGHEST_PROTOCOL)
#

# viz_sequence('results/monte_carlo_Q_test.pkl')
#
# with open('results/monte_carlo_stats_test.pkl', 'rb') as file:
#     stats = pickle.load(file)
#
# plot_episode_stats(stats,smoothing_window=20000)