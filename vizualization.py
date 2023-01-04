import pickle
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc
import matplotlib.pyplot as plt
import numpy as np


def viz(V, fig_title):
    Y, X = np.mgrid[range(12, 22), range(2, 12)]
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, V, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    title = "Value Function " + fig_title
    plt.title(title)
    plt.ylabel('Player Sum')
    plt.xlabel('Dealer Sum')
    plt.show()


def policy_vis(Q,fig_title):
    val1 = ["{:d}".format(i) for i in range(12, 22)]
    val2 = ["{:d}".format(i) for i in range(2, 12)]
    val3 = [[] for i in range(10)]
    color= [[] for i in range(10)]
    for d in range(10):
        for p in range(10):
            str = ""
            if np.argmax(Q[p][d]) == 1:
                str = "Hit"
                color[p].append("#aff7b4")
            else:
                str = "Stick"
                color[p].append("#f7afb8")
            val3[p].append(str)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = ax.table(
        cellText=val3,
        rowLabels=val1,
        colLabels=val2,
        rowColours=["palegreen"] * 10,
        colColours=["palegreen"] * 10,
        cellColours=color,
        cellLoc='center',
        loc='upper left')
    ax.set_title(fig_title,
                 fontweight="bold")
    plt.show()

def viz_sequence(File):
    with open(File, 'rb') as file:
        Q = pickle.load(file)
    Q_ace = np.zeros((22, 12, 2))
    Q_no_ace = np.zeros((22, 12, 2))
    for s1 in range(12, 22):
        for s2 in range(2, 12):
            state_ace = (s1, s2, 1)
            state_no_ace = (s1, s2, 0)
            Q_ace[s1][s2] = Q[state_ace]
            Q_no_ace[s1][s2] = Q[state_no_ace]
    Q_ace = Q_ace[12:22, 2:12]
    Q_no_ace = Q_no_ace[12:22, 2:12]
    V_ace = np.max(Q_ace, axis=2)
    V_no_ace = np.max(Q_no_ace, axis=2)
    viz(V_ace, "When The Player Has An Ace")
    viz(V_no_ace, "When The Player Has No Ace")
    policy_vis(Q_no_ace, fig_title="Optimal Policy When The Player Has No Ace")
    policy_vis(Q_ace, fig_title="Optimal Policy When The Player Has Ace")


def viz_sequence_Q(Q):
    Q_ace = np.zeros((22, 12, 2))
    Q_no_ace = np.zeros((22, 12, 2))
    for s1 in range(12, 22):
        for s2 in range(2, 12):
            state_ace = (s1, s2, 1)
            state_no_ace = (s1, s2, 0)
            Q_ace[s1][s2] = Q[state_ace]
            Q_no_ace[s1][s2] = Q[state_no_ace]
    Q_ace = Q_ace[12:22, 2:12]
    Q_no_ace = Q_no_ace[12:22, 2:12]
    V_ace = np.max(Q_ace, axis=2)
    V_no_ace = np.max(Q_no_ace, axis=2)
    viz(V_ace, "When The Player Has An Ace")
    viz(V_no_ace, "When The Player Has No Ace")
    policy_vis(Q_no_ace, fig_title="Optimal Policy When The Player Has No Ace")
    policy_vis(Q_ace, fig_title="Optimal Policy When The Player Has Ace")


def Q_to_array(Q):
    Q_arr = np.zeros((22, 12, 2))
    for s1 in range(12, 22):
        for s2 in range(2, 12):
            state = (s1, s2, 1)
            Q_arr[s1][s2] = Q[state]
    Q_arr = Q_arr[12:22, 2:12]
    return Q_arr

def Q_to_array_double(Q):
    Q_arr = np.zeros((22, 12, 3))
    for s1 in range(4, 22):
        for s2 in range(2, 12):
            state = (s1, s2, 1)
            Q_arr[s1][s2] = Q[state]
    Q_arr
    return Q_arr


def viz_sequence_double(File):
    with open(File, 'rb') as file:
        Q = pickle.load(file)
    Q_ace = np.zeros((22, 12, 3))
    Q_no_ace = np.zeros((22, 12, 3))
    for s1 in range(4, 22):
        for s2 in range(2, 12):
            state_ace = (s1, s2, 1)
            state_no_ace = (s1, s2, 0)
            Q_ace[s1][s2] = Q[state_ace]
            Q_no_ace[s1][s2] = Q[state_no_ace]
    Q_ace = Q_ace[4:22, 2:12]
    Q_no_ace = Q_no_ace[4:22, 2:12]
    V_ace = np.max(Q_ace, axis=2)
    V_no_ace = np.max(Q_no_ace, axis=2)
    viz_double(V_ace, "When The Player Has An Ace")
    viz_double(V_no_ace, "When The Player Has No Ace")
    policy_vis_double(Q_no_ace, fig_title="Optimal Policy When The Player Has No Ace")
    policy_vis_double(Q_ace, fig_title="Optimal Policy When The Player Has Ace")

def viz_double(V, fig_title):
    Y, X = np.mgrid[range(4, 22), range(2, 12)]
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, V, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    title = "Value Function " + fig_title
    plt.title(title)
    plt.ylabel('Player Sum')
    plt.xlabel('Dealer Sum')
    plt.show()

def policy_vis_double(Q,fig_title):
    val1 = ["{:d}".format(i) for i in range(4, 22)]
    val2 = ["{:d}".format(i) for i in range(2, 12)]
    val3 = [[] for i in range(18)]
    color= [[] for i in range(18)]
    for d in range(10):
        for p in range(18):
            str = ""
            if np.argmax(Q[p][d]) == 1:
                str = "Hit"
                color[p].append("#aff7b4")
            elif np.argmax(Q[p][d]) == 0:
                str = "Stick"
                color[p].append("#f7afb8")
            else:
                str = "Double"
                color[p].append("#92cbdf")
            val3[p].append(str)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    print(val1)
    table = ax.table(
        cellText=val3,
        rowLabels=val1,
        colLabels=val2,
        rowColours=["palegreen"] * 20,
        colColours=["palegreen"] * 10,
        cellColours=color,
        cellLoc='center',
        loc='upper left')
    ax.set_title(fig_title,
                 fontweight="bold")
    plt.show()

