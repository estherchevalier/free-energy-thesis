#!/usr/bin/env python3

import numpy as np

def remove_zeros(arr):

    if type(arr) == int:
        return arr + np.exp(-16)

    arr = np.array(arr)
    arr = np.where(arr == 0, arr + np.exp(-16), arr)

    return arr


def softmax(x, T=1):
    """
    x: (nd-array)   elements to
    T: (int)        Temperature
    """
    x = np.array(x)

    if x.ndim == 1:
        return np.exp(x/T) / np.sum(np.exp(x/T))
    else:
        return np.exp(x/T) / np.sum(np.exp(x/T), axis=1)[:, None]

class ShortTerm:


    def __init__(self, agent, reliability_prior):

        self.agent = agent

        # Prior belief about environment
        self.r = self.agent.r


    def __str__(self):

        pres = "\n#### Short Term ####\n"
        hint = "Belief about hint reliability:\n" + str(self.r) + '\n'

        return pres + hint

    def reset(self, reliability_prior):
        # Prior belief about environment
        self.belief_hint_reliability = np.array([reliability_prior, 1-reliability_prior])

    def belief_updating(self, A, r, hint):

        if hint > 0:

            A = remove_zeros(A)
            factor = (A[1]+ A[2])
            p = A[1] / factor

            # If reliability prior = 0, then hint accuracy is 50%
            reliability = remove_zeros([r/2 + .5, 1-(r/2 + .5)])

            # If reliability prior = 0, then hint accuracy is 0%
            # reliability = remove_zeros([r, 1-r])

            reliability = reliability[hint-1]

            p = (reliability * p) / ((reliability * p) + (1-reliability) * (1-p))

            # A[1] = p * factor
            # A[2] = factor - A[1]

            A = np.array([0, p, 1-p, 0])

        return A


if __name__ == "__main__":

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    from Agent import Agent
    from Environment import Environment

    env = Environment()
    a = Agent(env)
    st = a.short_term

    print(st.belief_updating([.2, .3, .1, .4], .98, 2))
