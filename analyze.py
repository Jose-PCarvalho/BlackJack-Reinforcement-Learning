from vizualization import *
from plotting import *
import dill
#viz_sequence('results/q_lambda.pkl')

with open('results/double_q_learning_stats.pkl', 'rb') as file:
    double_q_stats = pickle.load(file)

with open('results/monte_carlo_stats.pkl', 'rb') as file:
    monte_carlo_stats = pickle.load(file)

with open('results/q_lambda_stats.pkl', 'rb') as file:
    q_lambda_stats = pickle.load(file)

with open('results/q_learning_stats.pkl', 'rb') as file:
    q_stats = pickle.load(file)

with open('results/sarsa_lambda_stats.pkl', 'rb') as file:
    sarsa_lambda_stats = pickle.load(file)

with open('results/sarsa_stats.pkl', 'rb') as file:
    sarsa_stats = pickle.load(file)

double_q_rewards = pd.Series(double_q_stats.episode_rewards).rolling(1000000, min_periods=1000000).mean()

print("Double Q Mean-Reward ", double_q_rewards.iat[-1])

monte_carlo_rewards = pd.Series(monte_carlo_stats.episode_rewards).rolling(1000000, min_periods=1000000).mean()
print("Monte Carlo Mean-Reward ", monte_carlo_rewards.iat[-1])

q_lambda_rewards = pd.Series(q_lambda_stats.episode_rewards).rolling(1000000, min_periods=1000000).mean()
print("Q Lambda Mean-Reward ", q_lambda_rewards.iat[-1])

q_rewards = pd.Series(q_stats.episode_rewards).rolling(1000000, min_periods=1000000).mean()
print("Q Mean-Reward ", q_rewards.iat[-1])

sarsa_lambda_rewards = pd.Series(sarsa_lambda_stats.episode_rewards).rolling(1000000, min_periods=1000000).mean()
print("SARSA Lambda Mean-Reward ", sarsa_lambda_rewards.iat[-1])

sarsa_rewards = pd.Series(sarsa_stats.episode_rewards).rolling(1000000, min_periods=1000000).mean()
print("SARSA Mean-Reward ", sarsa_rewards.iat[-1])

