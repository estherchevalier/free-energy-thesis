import numpy as np


import pandas as pd

from Environment import Environment
from LongTerm import LongTerm
from ShortTerm import ShortTerm
from Agent import Agent

import matplotlib.pyplot as plt
import matplotlib.colors as mco
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


from itertools import product, cycle
from sklearn.model_selection import ParameterGrid

import warnings
warnings.filterwarnings("ignore")

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



# Only take action sequences that are possible considering absorbing states
policies = np.array([
                    # (0, 0),
                    # (0, 1),
                    # (0, 2),
                    # (0, 3),
                    # (3, 0),
                    (1, 1),
                    (3, 1),
                    # (1, 0),
                    # (1, 2),
                    # (1, 3),
                    (2, 2),
                    (3, 2),
                    # (2, 0),
                    # (2, 1),
                    # (2, 3),
                    #(3, 3),
                    ])

actions_dict = {(1, 1): 0, (2, 2): 1, (3, 1): 2, (3, 2): 3, (3, 3): 4}
my_dict = {0: "LL", 1: "RR", 2: "HM", 3: "HL", 4: "HR", 5: "HH", 6: "MM"}

# print(policies)



def expected_reward(step=80, loops=1, t=0.1):

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    x = np.zeros((6, step+1, step+1))

    true_rewards = np.zeros((step+1, step+1))
    step = (1 / step)

    i = 0

    for r in np.arange(1 + step, step=step):

        j = 0

        for p in np.arange(1 + step, step=step):

            env = Environment(food_is_right=p, hint_reliability=r)
            a = Agent(env, T=t, food_is_right_prior=p, reliability_prior=r)
            exp_out_total = np.zeros(6)
            t_reward = 0

            for _ in range(loops):

                a.long_term.policies = np.concatenate((policies,
                                                np.array([[3, env.hint], [3, 3-env.hint]])))
                #print(a.long_term.policies)
                true_reward, out = a.episode()
                out = np.sum(out, axis=0)

                exp_out_total += out
                t_reward += true_reward

                a.reset(a.A, a.r)

            exp_out_total /= loops
            t_reward /= loops

            x[:, i, j] = exp_out_total
            true_rewards[i, j] = t_reward

            j += 1

        i += 1

    max_policy = np.argmax(x, axis=0)
    expected_rewards = np.amax(x, axis=0)

    # m = plt.imshow(max, origin='lower', extent=ex, cmap=my_cmap)

    return max_policy, true_rewards, expected_rewards



def plot_accuracy(acc_inf, acc_learn, ps):

    # https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib

    fig, ax = plt.subplots(1)
    ax.plot(acc_inf, label="Active inference")
    ax.plot(acc_learn, label="Learning")
    ax.plot(np.arange(EPOCHS), ps[:EPOCHS], 'r--', label="Expected accuracy \n(optimal policy)")
    #ax.plot(np.arange(EPOCHS, (2*EPOCHS)), ps[EPOCHS:], 'r--')
    ax.plot(np.arange(EPOCHS, (2*EPOCHS)), [1] * EPOCHS, 'r--')
    #plt.axhline(y=, color='r', linestyle='--', label="Expected reward optimal policy")


    plt.ylim([0, 1.1])
    plt.xticks(np.arange(0, len(acc_inf), 1), np.arange(0, len(acc_inf), 1))
    plt.yticks(np.arange(0, 1.2, .2))

    plt.grid(axis='y')

    ax.annotate("Test", xy=(EPOCHS / 4, -1))
    #r"$p_{True} = 0.9$\n$r_{True} = 0.9$"
#    , xytext=(0, 5),xycoords='axes fraction', textcoords='offset points',
    #        size='large', ha='center', va='baseline', fontsize=12)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("Accuracy over epochs ("+str(EPISODES)+" runs per epoch)")
    plt.show()

def plot_As(As):

    ylim = [0, 1]

    if len(As) % 2 == 0:
        x, y = (int(len(As) / 2), int(2))

    if len(As) % 4 == 0:
        x, y = (int(len(As) / 4), 4)

    else:
        x, y = (len(As), 1)

    fig, axs = plt.subplots(x, y)

    for i, (A, ax) in enumerate(zip(As, axs.reshape(-1))):

        ax.bar(np.arange(4), A)
        ax.set_title("Epoch " + str(i))
        ax.set_ylim(ylim)
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(0, 1.2, 0.2))


    plt.tight_layout()
    plt.show()



def true_best_policy(step=5, loops=3):

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    x = np.zeros((6, step+1, step+1))
    y = np.zeros((step+1, step+1))
    step = (1 / step)

    i = 0

    param_grid = {'p': np.arange(1 + step, step=step), 'r' : np.arange(1 + step, step=step)}
    grid = ParameterGrid(param_grid)

    for params in grid:

        env = Environment(food_is_right=params['p'], hint_reliability=params['r'])

        for i, pol in enumerate(policies):

            a = Agent(env, T=.01, food_is_right_prior=params['p'], reliability_prior=params['r'],policy=[pol])
            true_reward = 0

            for _ in range(loops):

                true_reward += a.episode()[0]
                a.reset(a.A, a.r)

            true_reward /= loops
            x[i, int(params['r']*(1/step)), int(params['p']*(1/step))] = true_reward

        true_reward = 0
        #print(x[:4])

        for _ in range(loops):

            hint = env.hint
            a.long_term.policies = np.array([[3, hint]])
            true_reward += a.episode()[0]
            a.reset(a.A, a.r)

        true_reward /= loops
        x[4, int(params['r']*(1/step)), int(params['p']*(1/step))] = true_reward

        true_reward = 0

        for _ in range(loops):

            hint = env.hint
            a.long_term.policies = np.array([[3, 3-hint]])
            true_reward += a.episode()[0]
            a.reset(a.A, a.r)

        true_reward /= loops
        x[5, int(params['r']*(1/step)), int(params['p']*(1/step))] = true_reward

        true_reward = 0

        for _ in range(loops):

            hint = env.hint
            a.long_term.policies = policies
            true_reward += a.episode()[0]
            a.reset(a.A, a.r)

        true_reward /= loops
        y[int(params['r']*(1/step)), int(params['p']*(1/step))] = true_reward


    print(x)

    best_policy = np.argmax(x, axis=0)
    reward_best_policy = np.amax(x, axis=0)
    true_reward_gained = y


    return best_policy, reward_best_policy, true_reward_gained



def plot_expected_reward(step=80, loops=25):

    fig1, axs1 = plt.subplots(2,2, figsize=(12.8, 7.2))
    fig2, axs2 = plt.subplots(2, 2, dpi=130, sharey=True)
    fig1.suptitle("Policy with highest expected reward \ndepending on agent's prior beliefs and temperature", fontsize=14)
    fig2.suptitle("True and expected reward depending on \nagent's prior beliefs and temperature", fontsize=14)
    Ts = [0.01, 0.1, 1, 10]

    my_colorsc = np.array(plt.cm.tab20([0, 1, 6, 7, 8, 4]))
    my_cmap = mco.ListedColormap(my_colorsc)
    ex =[0,1,0,1]

    for i, t in enumerate(Ts):

        best_pol, true, expected = expected_reward(step=step, t=t, loops=loops)

        ax1 = axs1.reshape(-1)[i]
        m = ax1.imshow(best_pol, origin='lower', extent=ex, cmap=my_cmap, vmin=0, vmax=5)
        ax1.set_xlabel("p")

        ax1.set_title("Temperature " + str(t))

        if (i == 0) or (i == 3):

            if i ==3:
                d = 1
            else:
                d = 0

            ax21, ax22 = axs2[:, d]

            n = ax21.imshow(true, origin='lower', extent=ex, cmap='viridis', vmin=0, vmax=1)
            ax21.set_xlabel("p")
            ax21.set_xticks([0, .5, 1])
            ax21.set_yticks([0, .5, 1])

            ax21.annotate("Temperature\n" +str(t), xy=(.5, 1.1), xytext=(0, 5),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline', fontsize=10)

            #ax21.set_title("Temperature " + str(t), x=-2, y=0.6)

            o = ax22.imshow(expected, origin='lower', extent=ex, cmap='viridis', vmin=0, vmax=1)
            ax22.set_xlabel("p")
            ax22.set_xticks([0, .5, 1])
            ax22.set_yticks([0, .5, 1])

            #ax22.set_ylabel("r", rotation=0)
            #ax22.set_title("Temperature " + str(t))

    x1, x2 = axs2[:, 0]
    x1.set_title("True reward", x=-.65, y=.1, fontsize=10, rotation=90)
    x2.set_title("Expected reward", x=-.65, y=.1, fontsize=10, rotation=90)

    x1.set_ylabel("r", rotation=0)
    x2.set_ylabel("r", rotation=0)
    x1.set_xticks([0, .5, 1])
    x2.set_xticks([0, .5, 1])

    fig1.tight_layout()
    fig2.tight_layout()


    fig1.subplots_adjust(right=0.8)
    cbar_ax1 = fig1.add_axes([0.85, 0.15, 0.05, 0.7])

    #divider = make_axes_locatable()
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig2.colorbar(
        o,
        #cax=cax,
        ax=axs2.ravel().tolist(),
        extend="neither",
        spacing="uniform",
        ticks=np.arange(0,1.2,.2),
    )

    cbar.set_label('Average true/expected reward')

    tick_locs = np.arange(5/12, 5, 10/12)
    cbar = fig1.colorbar(m, cax=cbar_ax1, ticks=tick_locs)
    cbar_ax1.set_yticklabels(["Left", "Cue → Left", "Right", "Cue → Right", "Cue →\nCued position", "Cue →\nNot cued position"])
    cbar_ax1.set_ylabel('Expected best policy')

    fig1.savefig('../graphics/figures/expected_reward_'+ str(step)+"_steps_"+str(loops)+"_loops.svg", format='svg')
    fig2.savefig('../graphics/figures/diff_expected_reward_'+ str(step)+"_steps_"+str(loops)+"_loops.svg", format='svg')

    plt.show()

    return

def plot_true_best_policy(step=5, loops=3):

    fig, ax = plt.subplots(1, figsize=(15, 15))
    fig2, ax2 = plt.subplots(1, figsize=(15, 15))
    #fig.suptitle("Expected reward per policy", fontsize=14)
    #Ts = [0.01, 0.1, 1, 10]

    my_colorsc = np.array(plt.cm.tab20([0, 1, 6, 7, 8, 4]))
    my_cmap = mco.ListedColormap(my_colorsc)
    ex =[0,1,0,1]

    opt_pol, opt_reward, true_reward = true_best_policy(step=step, loops=loops)

    m = ax.imshow(opt_pol, origin='lower', extent=ex, cmap=my_cmap, vmin=0, vmax=5)
    ax.set_xlabel("p")
    ax.set_ylabel("r", rotation=0)
    #ax.set_title("Temperature " + str(t))

    plt.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    tick_locs = np.arange(5/12, 5, 10/12)
    cbar = fig.colorbar(m, cax=cbar_ax, ticks=tick_locs)

    cbar.ax.set_yticklabels(["Left", "Cue → Left", "Right", "Cue → Right", "Cue → Cued position", "Cue → Not cued position"])
    cbar_ax.set_ylabel('True best policy')

    fig.savefig('../graphics/figures/true_best_policy_'+ str(step)+"_steps_"+str(loops)+"_loops.svg", format='svg')

    m = ax2.imshow(np.abs(opt_reward-true_reward), origin='lower', extent=ex, cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel("p")
    ax.set_ylabel("r", rotation=0)
    #ax.set_title("Temperature " + str(t))

    plt.tight_layout()

    fig2.subplots_adjust(right=0.8)
    cbar_ax = fig2.add_axes([0.85, 0.15, 0.05, 0.7])

    #tick_locs = np.arange(5/12, 5, 10/12)
    cbar = fig2.colorbar(m, cax=cbar_ax)

    #cbar.ax.set_yticklabels(["Left", "Cue → Left", "Right", "Cue → Right", "Cue → Cued position", "Cue → Not cued position"])
    cbar_ax.set_ylabel('True best policy')

    fig2.savefig('../graphics/figures/diff_opt_reward_true_reward_'+ str(step)+"_steps_"+str(loops)+"_loops.svg", format="svg")
    plt.show()

    return

if __name__ == "__main__":

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    #print(expected_reward(5, 2))
    #plot_expected_reward(12, 1)
    plot_true_best_policy(step=15, loops=1)




    # for i, ax in enumerate(axs.reshape(-1)):
    #
    #     m = ax.imshow(x[i], origin='lower', vmin=0, vmax=1, extent=ex)
    #     ax.xaxis.set_ticks_position('bottom')
    #     ax.set_title(titles[i] + str(i))
    #
    # plt.tight_layout()
    #
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #
    # fig.colorbar(m, cax=cbar_ax)
    # cbar_ax.set_ylabel('Expected reward')
    # plt.show()



# colorName = ["LL", "RR", "HL", "HR"]
# colorVals = list(range(len(colorName)))
#
# my_colorsc = np.array(plt.cm.tab20c(list(range(16))))
# my_colorsc = np.array(plt.cm.tab20c([0, 1, 2, 3, 4, 8, 12, 13, 14, 15]))
#
# my_cmap = mco.ListedColormap(my_colorsc)
#
# import plotly.express as px
#
# step = 0.1
#
# fig, axs = plt.subplots(1, 2)
# ex = [0, 1, 0, 1]
#
# axs[0].set_xlabel("Prior belief food on the left")
# axs[0].set_ylabel("prior belief on hint reliablity")
#
# m0 = axs[0].imshow(
#     optimal_policy(step),
#     origin="lower",
#     cmap=my_cmap,
#     extent=ex,
#     aspect="auto",
#     vmin=0,
#     vmax=15,
# )
#
#
# m = axs[1].imshow(
#     plot_actual_policy(step),
#     origin="lower",
#     cmap=my_cmap,
#     extent=ex,
#     aspect="auto",
#     vmin=0,
#     vmax=15,
# )
#
#
# axs[0].set_title("Optimal policy")
# axs[1].set_title("Agent behaviour")
#
#
# divider = make_axes_locatable(axs[1])
# cax = divider.append_axes("right", size="5%", pad=0.1)
# cbar = fig.colorbar(
#     m,
#     cax=cax,
#     extend="neither",
#     spacing="uniform",
#     ticks=list(np.arange(10 / 15, 15, 10 / 15)),
# )
#
# # cbar.ax.set_yticklabels(list(map(str, policies)))
# # cbar.ax.set_yticklabels(list(np.arange(0,1,1/16)))
#
# plt.show()
