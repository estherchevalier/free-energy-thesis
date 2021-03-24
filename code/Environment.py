#!/usr/bin/env python3

import numpy as np
import pandas as pd

class Environment:

    def __init__(self, hint_shows_left=.75, hint_reliability=.98, hint=-1):

        if hint < 0:
            self.hint = np.random.choice([1, 2], p=[hint_shows_left, 1-hint_shows_left])
        else:
            self.hint = hint

        self.hint_shows_left = hint_shows_left

        # Probability of getting a reward at the cued location.
        # Cue is in location 3.
        r = hint_reliability
        self.reliability = np.array([r, 1-r])
        self.reliability = np.array([(1+r)*.5, (1-r)*.5])

        self.food_location = np.random.choice([self.hint, 3-self.hint], p=self.reliability)

        self.hint_visited = False

        self.true_agent_state = np.zeros(4)
        self.true_agent_state[0] = 1



        # p = 1 if self.food_location == 1 else -1
        # q = -1 if p == 1 else 1

        # p = 1 if self.food_location == 1 else 0
        # q = 0 if p == 1 else 1


        #self.true_A = np.array([0,p,q,0])
        self.true_A = np.zeros(4)
        self.true_A[self.food_location] = 1

        self.A_pos = np.identity(4)

        # Transition p(state at time t | state at t-1, action)

        # Probability of going from state at time t to state at time t+1
        # when performing action a.
        # shape: (4, 4, 4) = 4 actions, 4 possible states at time t,
        #                   4 possible states at time t+1

        ## x: Current action
        ## y: State at time t
        ## z: State at time t+1
        ## B[x, y, z]

        self.true_B = np.array([[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0]],

                            [[0,1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0]],

                            [[0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 1, 0]],

                            [[0, 0, 0, 1],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]])


    def __str__(self):

        pres = "\n#### Environment ####\n"
        s = "Food is at:" + str(self.food_location)
        h =  "    Hint shows:" + str(self.hint)
        r = "\nTrue reliability:" + str(self.reliability)
        fl = "\Hint shows left probability:" + str(self.hint_shows_left)


        return pres + s + h+ r + fl

    def reset(self):

        self.hint = np.random.choice([1, 2], p=[self.hint_shows_left, 1-self.hint_shows_left])
        self.food_location = np.random.choice([self.hint, 3-self.hint], p=self.reliability)
        self.true_A = np.zeros(4)
        self.true_A[self.food_location] = 1
        self.true_agent_state = np.array([1,0,0,0])
        self.hint_visited = False

    def execute_action(self, action):
        """
        """
        # Change true position of agent
        self.true_agent_state = self.true_agent_state @ self.true_B[action]
        reward = self.reward()
        state = np.argmax(self.true_agent_state)

        hint = -1

        if (state == 3):
            hint = self.hint
            self.hint_visited = True

        current_state = self.true_agent_state

        return reward, hint, current_state



    def reward(self):

        reward = np.sum(self.true_agent_state @ self.true_A)

        return reward



    def render(self, A, reliability):

        #🐈
        #'✔ ' "⛁ "

        p = A[1]
        q = A[2]

        np.set_printoptions(formatter={'float': lambda x: "{:.2f}".format(x)})

        print('\n▛', end='')
        for _ in range(22):
            print('▔', end='')
        print('▜')

        left = f"        "
        right = f""

        state = np.argmax(self.true_agent_state)

        if (state == 1):

            if self.food_location == 1:

                left = "🧀 🐁  "
                right = ' '

            if self.food_location == 2:
                right = " 🧀"
                left = "🐁  ▕"

        if (state == 2):

            if self.food_location == 1:
                left = "🧀 "
                right = "▏  🐁"

            if self.food_location == 2:
                left = " "
                right = "  🐁 🧀"

        print('▒ ' + left +'            ' + right+' ▒')

        left = f"{p:.2f}"
        right = f"{q:.2f}"
        r = f"{reliability:.2f}"

        print('▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')
        print('▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')

        print('▒▒'+left+'▒▒▒▒    ▒▒▒▒'+right+'▒▒')

        if (state == 0):
            print('▒▒▒▒▒▒▒▒▒▒ 🐁 ▒▒▒▒▒▒▒▒▒▒')
        else:
            print('▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')
        print('▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')
        print('▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')

        if (state == 3):
            print('▒▒▒▒▒▒▒▒▒▒🐁  ▒▒▒▒▒▒▒▒▒▒')
        else:
            print('▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')


        if self.hint == 1 and self.hint_visited:
            #print('▙▁▁▁▁▁▁▁▁▁  L ▁▁▁▁▁▁▁▁▁▟')
            print('▒▒▒▒▒▒▒▒▒▒  L ▒▒▒▒▒▒▒▒▒▒')

        elif self.hint == 2 and self.hint_visited:
            #print('▙▁▁▁▁▁▁▁▁▁  R ▁▁▁▁▁▁▁▁▁▟')
            print('▒▒▒▒▒▒▒▒▒▒  R ▒▒▒▒▒▒▒▒▒▒')

        else:
            print('▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')

        print('▙▁▁▁▁▁▁▁▁▁'+r+'▁▁▁▁▁▁▁▁▁▟')
        #print('▙▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▟')
        print()


if __name__ == "__main__":

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
