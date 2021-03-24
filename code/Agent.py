#!/usr/bin/env python3

import numpy as np
import pandas as pd

from Environment import Environment
from LongTerm import LongTerm
from ShortTerm import ShortTerm



class Agent:

    def __init__(self, env, T=1, food_is_left_prior=.75,
                                reliability_prior=.98,
                                mode='inference'):


        #self.belief_food_location = np.array([1, food_is_left_prior, 1-food_is_left_prior, 1])
        self.p = food_is_left_prior
        self.r = reliability_prior
        self.A = np.array([0, self.p, 1-self.p, 0])

        # Prior belief about environment
        #self.belief_hint_reliability = np.array([self.reliability_prior, 1-self.reliability_prior])


        self.env = env
        self.long_term = LongTerm(self, food_is_left_prior)
        self.short_term = ShortTerm(self, reliability_prior)

        self.temperature = T

        self.time_step = 0
        self.hint = -1


        # Keeping track of past states and n_observations
        # Maybe logging more appropriate (see self.reset)
        self.past_states_all = list()
        self.past_states_current_episode = list()

        # At the beginning, the agent is at location 0
        # and is certain about it (=> q(location 0 | time step 0) = 1)
        self.start_state = np.array([[1,0,0,0],[1,0,0,0]])
        self.current_state = self.start_state

        self.info = False
        self.mode = mode


    def __str__(self):

        a = "\n---- Agent description ----\n"
        aa = "Current position:  " + str(self.current_state) + '\n'
        #food = "Belief about food location:\n" + str(self.belief_food_location) + '\n'
        epi = "(Action, reward) / episode" + str(self.past_states_current_episode)
        st = str(self.short_term)
        lt = str(self.long_term)
        env = str(self.env)
        t = "\n\nTemperature: " + str(self.temperature)

        b = "\n---- ----------------- ----\n"

        return a + st + lt + env + t + b




    def reset(self, A, r):

        self.env.reset()
        self.short_term.reset(r)

        self.time_step = 0
        self.hint = -1
        self.p = A[1]
        self.A = A
        self.start_state = np.array([[1,0,0,0],[1,0,0,0]])
        self.current_state = self.start_state

        # To optimize through learning
        # self.belief_food_location = np.array([1, self.food_is_left_prior, 1-self.food_is_left_prior, 1])
        self.long_term.reset(A)

        # Keeping track of past states and n_observations
        # TODO logging more appropriate
        self.past_states_all += self.past_states_current_episode
        self.past_states_current_episode = list()


        #self.temperature -= 1


    # def episode(self, mode="learning", info=True):
    #
    #     actions = []
    #     rewards = 0
    #
    #     if info:
    #         self.env.render(self.A, self.r)
    #
    #     while self.time_step < 2:
    #
    #         exp_reward, action, q_s = self.expected_reward()
    #         true_reward = self.execute_action(action)
    #         rewards += true_reward
    #
    #         if mode == "learning":
    #
    #             A = self.learn(q_s, true_reward)
    #             self.long_term.A = A
    #
    #
    #         if info:
    #             self.env.render(self.long_term.A, self.r)
    #             print("Expected reward", exp_reward, "True reward", true_reward)
    #             #print(self)
    #
    #         actions.append(action)
    #         self.time_step += 1
    #
    #
    #
    #     if mode == "learning":
    #         self.reset(self.long_term.A, self.r)
    #     else:
    #         self.reset(np.array([0, self.p, 1-self.p, 0]), self.r)
    #
    #
    #     return rewards


    def episode(self, info=True):

        actions = []
        rewards = []
        reward = 0

        if info:
            self.env.render(self.A, self.r)

        while self.time_step < 2:

            if self.mode == "learning":

                ### Missing: absorbing states

                q_s = self.long_term.sample_states()
                o = self.long_term.sample_outcomes(q_s)

                A = self.learn(q_s, o[:, self.time_step+1])
                # numbers_pol = self.long_term.policies.shape[0]
                # rewards = np.tile(reward, numbers_pol)
                #A = self.learn(q_s, rewards)
                self.A = A

                if info:
                    print("After gradient descent:")
                    self.env.render(self.A, self.r)


            self.long_term.A = self.A
            q_s = self.long_term.sample_states()
            o = self.long_term.sample_outcomes(q_s)

            efe = self.long_term.exp_free_energy_all_policies(q_s, o)

            q_pi = self.long_term.bayesian_averaging(efe)
            action  = self.action_selection(q_pi)

            true_reward = self.execute_action(action)
            exp_reward = self.long_term.expected_reward(q_s, action)

            reward += true_reward




            if info:
                self.env.render(self.long_term.A, self.r)
                print("Expected reward", exp_reward, "True reward", true_reward)
                #print(self)

            actions.append(action)
            self.time_step += 1

        if self.mode == "learning":
            self.reset(self.long_term.A, self.r)
        else:
            self.reset(np.array([0, self.p, 1-self.p, 0]), self.r)


        return reward



    def epoch(self, n_episodes, info=False):

        rewards = list()

        for i in range(n_episodes):
            print("###### Episode " + str(i+1) + " ####")

            if i % 10 == 0:
                r = self.episode(info=info)
            else:
                r = self.episode()

            rewards.append(r)
            #print(rewards)

        accuracy = np.count_nonzero(rewards) / n_episodes
        print("Accuracy:", accuracy)

        return accuracy, self.A


    def learn(self, q_states, exp_reward):

        #print("##### LEARNING #########")

        As = list()

        for i, (q_s, o) in enumerate(zip(q_states, exp_reward)):

            policy = self.long_term.policies[i]
            start_F = self.long_term.variational_free_energy(policy, q_s, o)

            A = self.long_term.gradient_descent(start_F, policy, q_s, o)

            #print("A", A, "Policy:", policy)
            As.append(A)

        As = np.array(As)

        print(As)

        A = np.average(As, axis=0, weights=self.long_term.q_pi)
        #print("New A", A)

        return A


    def action_selection(self, q_u, mode="sample_action"):

        if mode == 'sample_action':
            chosen_action = np.random.choice(self.long_term.actions, p=q_u)

        if mode == 'best_action':
            chosen_action = self.long_term.actions[np.argmax(q_u)]

        return chosen_action


    def execute_action(self, action):

        reward, hint, current_state = self.env.execute_action(action)

        if hint > 0:

            self.long_term.visited_hint = True
            self.current_state = self.current_state[hint-1]
            #self.long_term.p_obs = self.long_term.p_obs[hint-1]

            self.hint = hint
            self.A = self.belief_updating(hint)
            self.long_term.reset(self.A)

        return reward

    def expected_reward(self, info=False):

        fe, q_s = self.long_term.exp_free_energy_all_policies()
        q_u, q = self.long_term.bayesian_averaging(fe, q_s)
        action = self.action_selection(q_u)

        if info:
            print("Free energy per policy", fe)
            print("Expected reward over all\n", q, "\n")
            print("## Action taken ##", action)

        exp_reward = self.long_term.expected_reward(q, action)

        return exp_reward, action, q_s

    # def expected_reward(self, info=False):
    #
    #     fe, q_s = self.long_term.exp_free_energy_all_policies()
    #     q_u, q = self.long_term.bayesian_averaging(fe, q_s)
    #     action = self.action_selection(q_u)
    #
    #     if info:
    #         print("Free energy per policy", fe)
    #         print("Expected reward over all\n", q, "\n")
    #         print("## Action taken ##", action)
    #
    #     exp_reward = self.long_term.expected_reward(q, action)
    #
    #     return exp_reward, action, q_s


    def belief_updating(self, hint):

        A = self.short_term.belief_updating(self.A, self.r, hint)

        return A



if __name__ == "__main__":

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    env = Environment()
    a = Agent(env)

    print(a.execute_action(2))



    states = np.array([[[1,0,0,0],[1,0,0,0]],
                        [[0,1,0,0],[0,1,0,0]],
                        [[0,0,1,0],[0,0,1,0]],
                        [[0,0,0,1],[0,0,0,1]]]
                        )

    #states = np.identity(4)


    # for s in states:
    #     print("State", s)
    #     for action in range(4):
    #         fe = a.long_term.expected_free_energy(s, action, info=True)
    #         print('\n\n')
