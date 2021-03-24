from Agent import Agent
from Environment import Environment

import numpy as np

# from bokeh.plotting import figure, output_file, show
# from bokeh.layouts import row

import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

EPISODES = 5
EPOCHS = 11
MODE = "inference" # "learning" or "inference"


def plot(past_states_all):

    past_states_all = np.array(past_states_all)

    epi = np.arange(EPISODES) + 1
    rewards = past_states_all[1::2,2]
    food_loc = past_states_all[1::2,3]
    agent_loc = past_states_all[1::2,0]


    # output to static HTML file
    output_file("plots.html")

    # create a new plot with a title and axis labels
    p = figure(title="Rewards", x_axis_label='Episodes', y_axis_label='Reward')
    p.circle(epi, rewards, legend_label="Temp.", size=5)

    s3 = figure(background_fill_color="#fafafa")
    s3.square(epi, food_loc, size=12, color="#d95b43", alpha=0.8)
    s3.circle(epi, agent_loc, size=12, color="#53777a", alpha=0.5)

    # show the results
    show(row(p, s3))

actions_dict = {(1,1):0, (1,2):8, (1,3):9, (1,0):12,
                (3,0):11, (3,1):2,  (3,2):3, (3,3):4,
                (2,0):13, (2,1):6, (2,2):1, (2,3):5,
                (0,0):10, (0,1):15}


def plot_accuracy(acc):

    plt.plot(acc)
    plt.ylim([0, 1])
    plt.xticks(np.arange(len(acc)), np.arange(len(acc))+1)
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


def main():

    env = Environment(hint_shows_left=0.2)
    a = Agent(env, T=1, food_is_left_prior=0.5, reliability_prior=1, mode=MODE)

    accuracies = list()
    As = [a.A]

    for i in range(EPOCHS):

        print("######  Epoch " + str(i+1) + "  ####")
        acc, A = a.epoch(EPISODES, info=False)
        #print("Accuracy", acc, "A", A)
        accuracies.append(acc)
        As.append(A)


    print("Accuracies", accuracies)
    plot_accuracy(accuracies)
    plot_As(As)


main()
