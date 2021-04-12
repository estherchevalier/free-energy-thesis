from Agent import Agent
from Environment import Environment

from plotting import plot_As, plot_accuracy
import numpy as np

import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

EPISODES = 20
EPOCHS = 30
TEMP = 3 # temperature
#MODE = "inference" #
MODE = "learning"

def simple():

    env = Environment(food_is_right=1)
    a = Agent(env, T=1, food_is_right_prior=1, policy=[[2,2]], mode=MODE)

    for _ in range(EPISODES):
        a.episode(info=True)

def main():


    true_ps = [.9, .1]
    true_rs = [.3, 1]
    prior_p = .9
    prior_r = .3

    env = Environment(food_is_right=true_ps[0], hint_reliability=true_rs[0])

    act_inf = Agent(env, T=TEMP, food_is_right_prior=prior_p, reliability_prior=prior_r, mode="inference")
    learn = Agent(env, T=TEMP, food_is_right_prior=prior_p, reliability_prior=prior_r, mode="learning")

    #params = (true_p, true_r, prior_p, prior_r)


    accuracies_inf = list()
    accuracies_learn = list()
    ps = list()
    #As = [a.A]

    for true_r, true_p in zip(true_rs, true_ps):

        env = Environment(food_is_right=true_p, hint_reliability=true_r)

        act_inf.env = env
        learn.env = env

        for i in range(EPOCHS):

            if i % 20 == 0:
                print("######  Epoch " + str(i+1) + "  ####")

            acc_inf, A = act_inf.epoch(EPISODES)
            acc_learn, A = learn.epoch(EPISODES)
            #print("Accuracy", acc, "A", A)
            accuracies_inf.append(acc_inf)
            accuracies_learn.append(acc_learn)
            ps.append(true_p if true_p > .5 else 1- true_p)

        #As.append(A)


    #print("Accuracies", accuracies)
    plot_accuracy(accuracies_inf, accuracies_learn, ps, EPOCHS, EPISODES)
    #plot_As(As)

simple()
#main()
