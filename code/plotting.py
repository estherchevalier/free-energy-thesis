import numpy as np

# import bokeh as bk
import pandas as pd

from Environment import Environment
from LongTerm import LongTerm
from ShortTerm import ShortTerm
from Agent import Agent

import matplotlib.pyplot as plt
import matplotlib.colors as mco
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# from bokeh.plotting import figure, show
# from bokeh.transform import factor_cmap

from itertools import product, cycle


def belief_updating(p, reliability):

    p = (reliability * p) / ((reliability * p) + (1 - reliability) * (1 - p))

    return p


def reward(policy, p, r):
    """
    Expected reward if the hint shows to the left side of the maze.
    """

    reward = 0
    A = [0, p, 1 - p, 0]

    for a in policy:

        reward += A[a]

        if a == 3:
            p = belief_updating(p, r)
            A = [0, p, 1 - p, 0]

        else:
            A = np.zeros(4)

    return reward


# policies = np.array([p for p in product(np.arange(4), repeat=2)])
policies = [[1, 1], [2, 2], [3, 1], [3, 2], [3, 3]]

policies = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
]

policies = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (2, 2),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
]

actions_dict = {(1, 1): 0, (2, 2): 1, (3, 1): 2, (3, 2): 3, (3, 3): 4}
my_dict = {0: "LL", 1: "RR", 2: "HM", 3: "HL", 4: "HR", 5: "HH", 6: "MM"}

# print(policies)


def optimal_policy(step=0.01):

    mat = list()

    for pol in policies:

        policy = list()

        for r in np.arange(1 + step, step=step):

            food_left = list()

            for p in np.arange(1 + step, step=step):

                food_left.append(reward(pol, p, r))

            policy.append(food_left)

        mat.append(policy)

    np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

    mat = np.array(mat)
    phase = np.argmax(mat, axis=0)
    print(np.unique(phase))
    return phase


def plot_actual_policy(step=0.01):

    actions_dict = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 0): 4,
        (1, 1): 5,
        (1, 2): 6,
        (1, 3): 7,
        (2, 0): 8,
        (2, 1): 9,
        (2, 2): 10,
        (2, 3): 11,
        (3, 0): 12,
        (3, 1): 13,
        (3, 2): 14,
        (3, 3): 15,
    }

    actions_dict = {
        (0, 0): 0,
        (0, 1): 1,
        (0, 2): 2,
        (0, 3): 3,
        (1, 1): 4,
        (2, 2): 5,
        (3, 0): 6,
        (3, 1): 7,
        (3, 2): 8,
        (3, 3): 9,
    }

    # actions_dict = {(1,1):0,(2,2):1,
    #                 (3,1):2,  (3,2):3, (3,3):4}

    mat = list()

    for r in np.arange(1 + step, step=step):

        food_left = list()

        for p in np.arange(1 + step, step=step):

            env = Environment(hint_shows_left=1)

            a = Agent(env, T=0.2, food_is_left_prior=p, reliability_prior=r)
            act = actions_dict[tuple(a.episode())]
            food_left.append(act)

        mat.append(food_left)

    mat = np.array(mat)
    # b = mat.astype(str)

    return mat


colorName = ["LL", "RR", "HL", "HR"]
colorVals = list(range(len(colorName)))

my_colorsc = np.array(plt.cm.tab20c(list(range(16))))
my_colorsc = np.array(plt.cm.tab20c([0, 1, 2, 3, 4, 8, 12, 13, 14, 15]))

my_cmap = mco.ListedColormap(my_colorsc)

import plotly.express as px

step = 0.1

fig, axs = plt.subplots(1, 2)
ex = [0, 1, 0, 1]

axs[0].set_xlabel("Prior belief food on the left")
axs[0].set_ylabel("prior belief on hint reliablity")

m0 = axs[0].imshow(
    optimal_policy(step),
    origin="lower",
    cmap=my_cmap,
    extent=ex,
    aspect="auto",
    vmin=0,
    vmax=15,
)


m = axs[1].imshow(
    plot_actual_policy(step),
    origin="lower",
    cmap=my_cmap,
    extent=ex,
    aspect="auto",
    vmin=0,
    vmax=15,
)


axs[0].set_title("Optimal policy")
axs[1].set_title("Agent behaviour")


divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(
    m,
    cax=cax,
    extend="neither",
    spacing="uniform",
    ticks=list(np.arange(10 / 15, 15, 10 / 15)),
)

# cbar.ax.set_yticklabels(list(map(str, policies)))
# cbar.ax.set_yticklabels(list(np.arange(0,1,1/16)))

plt.show()
