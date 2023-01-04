import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "wins"])


def plot_episode_stats(stats, smoothing_window=2000, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    wins_smoothed = pd.Series(stats.wins).rolling(smoothing_window, min_periods=smoothing_window).mean()*100
    plt.plot(wins_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (%)")
    plt.title("Win Rate over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig1)
    else:
        plt.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show()

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show()

    return fig1, fig2, fig3

def plot_monte_carlo(stats_list,labels, smoothing_window=2000, noshow=False):
    fig2 = plt.figure(figsize=(10, 5))
    for stats, label in zip(stats_list,labels):
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed,label=label)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Average Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.legend(loc="lower right")
    if noshow:
        plt.close(fig2)
    else:
        plt.show()


def plot_question3(stats_list,labels, smoothing_window=2000, noshow=False):
    fig2 = plt.figure(figsize=(10, 5))
    for stats, label in zip(stats_list,labels):
        stat=stats.episode_rewards
        rewards_smoothed = pd.Series(stat).rolling(int(smoothing_window), min_periods=int(smoothing_window)).mean()
        plt.plot(rewards_smoothed,label=label)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Average Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.legend(loc="lower right")
    if noshow:
        plt.close(fig2)
    else:
        plt.show()

def plot_mse(stats_list,labels,title):
    fig2 = plt.figure(figsize=(10, 5))
    i=0
    for stats, label in zip(stats_list,labels):
        stat=stats.episode_lengths[1000:-1]
        lines=plt.plot(stats.episode_lengths,label=label)
        if i>5:
            lines[0].set_linestyle('dashed')
        else:
            lines[0].set_linestyle('solid')
        i+=1
    plt.xlabel("Episode")
    plt.ylabel("MSE")
    plt.title(title+"Mean Square Error Of The State-Action Value Funtion")
    plt.legend(loc="upper right")
    plt.show()

def plot_MSE_vs_Lambda(stats_list):
    values_sarsa=[]
    values_Q=[]
    fig2 = plt.figure(figsize=(10, 5))
    i=0
    for stats in stats_list:
        if i>5:
            values_Q.append(stats.episode_lengths[-1])
        else:
            values_sarsa.append(stats.episode_lengths[-1])
        i+=1


    lambdas=[0,0.2,0.4,0.6,0.8,1]
    print(values_Q)
    print(values_sarsa)
    plt.plot(lambdas,values_Q,label='Q Learning')
    plt.plot(lambdas, values_sarsa, label='SARSA')
    plt.xlabel("Lambda")
    plt.ylabel("MSE")
    plt.title("Mean Square Error Of The State-Action Value Funtion vs Lambda @ Episode 1000")
    plt.legend(loc="upper right")
    plt.show()
