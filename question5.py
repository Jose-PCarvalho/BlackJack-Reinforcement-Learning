from vizualization import *
from environment import *
from environment import *
from algorithms import *
from plotting import *
import pickle
import dill
from collections import namedtuple


env = BlackJack(mode="double")
lambdas = [0, 0.2, 0.4, 0.6, 0.8, 1]
stats_Q=[]
stats_SARSA=[]
labels_q=[]
labels_sarsa=[]
with open('results/MSE/double_q_baseline.pkl', 'rb') as file:
    baseline = pickle.load(file)

for ld in lambdas:
    Q, stats = sarsa_lambda(env, num_episodes=30000, ld=ld,Q_baseline=baseline)
    stats_SARSA.append(stats)
    labels_sarsa.append("SARSA Lambda={}".format(ld))
    Q,stats = watkins_q(env, num_episodes=30000, ld=ld,Q_baseline=baseline)
    labels_q.append("Q Lambda={}".format(ld))
    stats_Q.append(stats)
for s in stats_Q:
    stats_SARSA.append(s)
for l in labels_q:
    labels_sarsa.append(l)
title="Dutch Traces With MC Baseline - "
plot_mse(stats_SARSA,labels_sarsa,title)

viz_sequence_double('results/MSE/double_q_baseline.pkl')