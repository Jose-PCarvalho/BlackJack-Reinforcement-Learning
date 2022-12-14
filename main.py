from environment import *
from algorithms import *

env=BlackJack()
#Q=monte_carlo(env,num_episodes=int(5e5))
#Q=sarsa_lambda(env,num_episodes=int(1e4))
Q = sarsa(env,num_episodes= int(5e5))