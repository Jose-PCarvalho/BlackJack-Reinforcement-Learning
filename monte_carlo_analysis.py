from vizualization import *
from plotting import *
import dill

with open('results/monte_carlo/monte_carlo_N0_100_ALPH_LIN_STATS', 'rb') as file:
    monte_carlo_N0_100_ALPH_LIN_STATS = pickle.load(file)

with open('results/monte_carlo/monte_carlo_N0_100_ALPH_EXP_STATS', 'rb') as file:
    monte_carlo_N0_100_ALPH_EXP_STATS = pickle.load(file)

with open('results/monte_carlo/monte_carlo_N0_10_ALPH_LIN_STATS', 'rb') as file:
    monte_carlo_N0_10_ALPH_LIN_STATS = pickle.load(file)

with open('results/monte_carlo/monte_carlo_N0_10_ALPH_EXP_STATS', 'rb') as file:
    monte_carlo_N0_10_ALPH_EXP_STATS = pickle.load(file)

with open('results/monte_carlo/monte_carlo_SQRT_ALPH_EXP_STATS', 'rb') as file:
    monte_carlo_SQRT_ALPH_EXP_STATS = pickle.load(file)

with open('results/monte_carlo/monte_carlo_SQRT_ALPH_LIN_STATS', 'rb') as file:
    monte_carlo_SQRT_ALPH_LIN_STATS = pickle.load(file)

stats = [monte_carlo_N0_100_ALPH_LIN_STATS, monte_carlo_N0_100_ALPH_EXP_STATS, monte_carlo_N0_10_ALPH_LIN_STATS,
         monte_carlo_N0_10_ALPH_EXP_STATS, monte_carlo_SQRT_ALPH_EXP_STATS, monte_carlo_SQRT_ALPH_LIN_STATS]

labels = ["Epsilon:N0=100 Alpha:Linear","Epsilon:N0=100 Alpha:Polynomial", "Epsilon: N0=10 Alpha:Linear", "Epsilon: N0=10 Alpha:Polynomial", "Epsilon: SQRT Alpha:Polynomial", "Epsilon: SQRT Aplha:Linear"]

plot_monte_carlo(stats,labels,smoothing_window=20000)


with open('results/monte_carlo/monte_carlo_N0_100_ALPH_LIN_Q', 'rb') as file:
    monte_carlo_N0_100_ALPH_LIN_Q = pickle.load(file)

with open('results/monte_carlo/monte_carlo_N0_100_ALPH_EXP_Q', 'rb') as file:
    monte_carlo_N0_100_ALPH_EXP_Q = pickle.load(file)

with open('results/monte_carlo/monte_carlo_N0_10_ALPH_LIN_Q', 'rb') as file:
    monte_carlo_N0_10_ALPH_LIN_Q = pickle.load(file)

with open('results/monte_carlo/monte_carlo_N0_10_ALPH_EXP_Q', 'rb') as file:
    monte_carlo_N0_10_ALPH_EXP_Q = pickle.load(file)

with open('results/monte_carlo/monte_carlo_SQRT_ALPH_EXP_Q', 'rb') as file:
    monte_carlo_SQRT_ALPH_EXP_Q = pickle.load(file)

with open('results/monte_carlo/monte_carlo_SQRT_ALPH_LIN_Q', 'rb') as file:
    monte_carlo_SQRT_ALPH_LIN_Q = pickle.load(file)


Q = [monte_carlo_N0_100_ALPH_LIN_Q, monte_carlo_N0_100_ALPH_EXP_Q, monte_carlo_N0_10_ALPH_LIN_Q,
         monte_carlo_N0_10_ALPH_EXP_Q, monte_carlo_SQRT_ALPH_EXP_Q, monte_carlo_SQRT_ALPH_LIN_Q]

for q in Q:
    viz_sequence_Q(q)