from vizualization import *
from plotting import *
import dill

with open('results/monte_carlo_stats_test.pkl', 'rb') as file:
    monte_carlo_stats = pickle.load(file)


with open('results/q_learning_stats.pkl', 'rb') as file:
    q_stats = pickle.load(file)

with open('results/sarsa_stats.pkl', 'rb') as file:
    sarsa_stats = pickle.load(file)


stats = [monte_carlo_stats,q_stats,sarsa_stats]

labels = ["Monte Carlo", "Q-Learning","SARSA"]

plot_question3(stats,labels,smoothing_window=100000)



with open('results/monte_carlo_Q.pkl', 'rb') as file:
    monte_carlo = pickle.load(file)

with open('results/q_learning.pkl', 'rb') as file:
    q_Q = pickle.load(file)

with open('results/sarsa.pkl', 'rb') as file:
    sarsa_Q = pickle.load(file)


Q = [monte_carlo, q_Q, sarsa_Q,]

for q in Q:
    viz_sequence_Q(q)