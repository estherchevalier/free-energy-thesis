#!/usr/bin/env python3

import numpy as np

class Environment:

    def __init__(self, food_is_right=.75, hint_reliability=.98):

        self.fil = 1 - food_is_right
        self.food_location = np.random.choice([1, 2], p=[self.fil, 1-self.fil])

        # Probability of getting a reward at the cued location.
        # Cue is in location 3.
        r = hint_reliability

        # When reliability is 0, cue is fiable 50% of the time
        self.reliability = np.array([(1+r)*.5, (1-r)*.5])

        # When reliability is 0, cue is fiable 0% of the time
        # (change ShortTerm if uncommented)
        # self.reliability = np.array([r, (1-r)])

        # Define which way the hint is pointing at depending on cheese Location
        # and the cue reliability
        self.hint = np.random.choice([self.food_location, 3-self.food_location],
                                        p=self.reliability)


        # Agent always starts at position 0
        self.true_agent_state = np.zeros(4)
        self.true_agent_state[0] = 1

        #### Causal structure of Environment ###

        # The positive reward is at the cheese location
        self.true_A = np.zeros(4)
        self.true_A[self.food_location] = 1

        # Transition p(state at time t | state at t-1, action)

        # Probability of going from state at time t to state at time t+1
        # when performing action a.
        # shape: (4, 4, 4) = 4 actions,
        #                   4 possible states at time t,
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

        # Used for rendering
        self.hint_visited = False


    def __str__(self):

        pres = "\n#### Environment ####\n"
        s = "Food is at:" + str(self.food_location)
        h =  "    Hint shows:" + str(self.hint)
        r = "\nTrue reliability:" + str(self.reliability)
        fl = "\nHint shows left probability:" + str(self.hint_shows_left)


        return pres + s + h+ r + fl

    def reset(self):

        self.food_location = np.random.choice([1, 2], p=[self.fil, 1-self.fil])
        self.hint = np.random.choice([self.food_location, 3-self.food_location],
                                        p=self.reliability)

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

        # There is only one cheese - if cheese location is reached,
        # the reward is gained only once
        if (state == 1) or (state == 2):
            self.true_A = np.zeros(4)

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

        print('\n\t▛', end='')
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

        # left = ' 1  '
        # right = '  2 '
        print('\t▒ ' + left +'            ' + right+' ▒')

        left = f"{p:.2f}"
        right = f"{q:.2f}"
        r = f"{reliability:.2f}"

        print('\t▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')
        print('\t▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')
        print('\t▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')

        print('\t▒▒'+left+'▒▒▒▒    ▒▒▒▒'+right+'▒▒')

        if (state == 0):
            print('\t▒▒▒▒▒▒▒▒▒▒ 🐁 ▒▒▒▒▒▒▒▒▒▒')
            #print('\t▒▒▒▒▒▒▒▒▒▒  0 ▒▒▒▒▒▒▒▒▒▒')

        else:
            print('\t▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')
        print('\t▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')
        print('\t▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')

        if (state == 3):
            print('\t▒▒▒▒▒▒▒▒▒▒ 🐁 ▒▒▒▒▒▒▒▒▒▒')
        else:
            print('\t▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')


        if self.hint == 1 and self.hint_visited:
            #print('▙▁▁▁▁▁▁▁▁▁  L ▁▁▁▁▁▁▁▁▁▟')
            print('\t▒▒▒▒▒▒▒▒▒▒  L ▒▒▒▒▒▒▒▒▒▒')

        elif self.hint == 2 and self.hint_visited:
            #print('▙▁▁▁▁▁▁▁▁▁  R ▁▁▁▁▁▁▁▁▁▟')
            print('\t▒▒▒▒▒▒▒▒▒▒  R ▒▒▒▒▒▒▒▒▒▒')

        else:
            print('\t▒▒▒▒▒▒▒▒▒▒    ▒▒▒▒▒▒▒▒▒▒')

        #print('\t▙▁▁▁▁▁▁▁▁▁'+r+'▁▁▁▁▁▁▁▁▁▟')
        print('\t▒▒▒▒▒▒▒▒▒▒'+r+'▒▒▒▒▒▒▒▒▒▒')
        #print('\t▒▒▒▒▒▒▒▒▒▒  3 ▒▒▒▒▒▒▒▒▒▒')
        #print('\t▙▁▁▁▁▁▁▁▁▁  3 ▁▁▁▁▁▁▁▁▁▟')
        print('\t▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔')
        print()


if __name__ == "__main__":

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    env = Environment()
    env.render([0, 0.5, 0.5, 0], .98)
