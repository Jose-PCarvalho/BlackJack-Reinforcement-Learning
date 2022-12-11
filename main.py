from environment import *
from algorithms import *

env=BlackJack()
#Q=monte_carlo(env,num_episodes=int(1e7))
Q=sarsa_lambda(env,num_episodes=int(1e6))