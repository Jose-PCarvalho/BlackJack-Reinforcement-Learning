from vizualization import viz_sequence
from environment import *
from environment import *
from algorithms import *
from plotting import *
import pickle
import dill
from collections import namedtuple

env = BlackJack()
lambdas = [0, 0.2, 0.4, 0.6, 0.8, 1]
stats_Q=[]
stats_SARSA=[]
labels_q=[]
labels_sarsa=[]
with open('results/MSE/MC_baseline.pkl', 'rb') as file:
    baseline = pickle.load(file)
for ld in lambdas:
    Q, stats = sarsa_lambda(env, num_episodes=1000, ld=ld,Q_baseline=baseline)
    stats_SARSA.append(stats)
    labels_sarsa.append("SARSA Lambda={}".format(ld))
    Q,stats = watkins_q(env, num_episodes=1000, ld=ld,Q_baseline=baseline)
    labels_q.append("Q Lambda={}".format(ld))
    stats_Q.append(stats)
for s in stats_Q:
    stats_SARSA.append(s)
for l in labels_q:
    labels_sarsa.append(l)
title="Accumulating Traces With Double Q Baseline - "
plot_mse(stats_SARSA,labels_sarsa,title)
plot_MSE_vs_Lambda(stats_SARSA)

