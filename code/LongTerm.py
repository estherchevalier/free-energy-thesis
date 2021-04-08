#!/usr/bin/env python3

import numpy as np
from itertools import product


##### Helper functions #####################

_SQRT2 = np.sqrt(2)

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """

    p = np.array(p)
    q = np.array(q)

    # Remove zeros
    p = np.where(p == 0, p + np.exp(-16), p)
    q = np.where(q == 0, q + np.exp(-16), q)

    return np.sum(p * np.log(p / q))

def remove_zeros(p):
    return p + np.exp(-16)

def H(p):
    # Ambiguity
    np.seterr(divide='ignore', invalid='ignore')
    p = np.array(p)
    H = -np.sum(np.where(p != 0, p * np.log(p), 0))

    return H


def mean_square_error(q, p):

    mse = np.average((q - p) ** 2)

    return mse


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


##########################################

class LongTerm:

    def __init__(self, a, policy):

### Fixed variables
        self.agent = a

        # All possible actions:
        # Action 0: go to location 1, middle of the maze
        # Action 1: go to location 2, left arm
        # Action 2: go to location 3, right arm
        # Action 3: go to location 4, bottom arm, location of hint
        self.actions = np.arange(4)

        if policy is None:
            # All possible action sequences in two time steps
            self.policies = np.array([p for p in product(self.actions, repeat=2)])

        else:
            # Manually initialize which policies to consider
            self.policies = np.array(policy)


## Environmental model

        self.q_pi = np.zeros(self.policies.shape[0]) + (1/self.policies.shape[0])

        # Prior preferences on observations p(outcomes)
        # shape: (2, 4) = 2 contexts, 4 possible states

        #self.p_obs = softmax([[0,3,-3,0],[0,-3,3,0]]
        self.p_obs = np.array([1, 0])


        p = self.agent.p

        # p(o|s)
        self.A = np.array([0, p, (1-p), 0])



        # Transition p(state at time t | state at t-1, action)

        # Probability of going from state at time t to state at time t+1
        # when performing action a.
        # shape: (4, 4, 4) = 4 actions, 4 possible states at time t,
        #                   4 possible states at time t+1

        ## x: Current action
        ## y: State at time t
        ## z: State at time t+1
        ## B[x, y, z]

        self.B = self.agent.env.true_B

        self.visited_hint = False


    def __str__(self):
         pres = "\n##### Long Term #####\n"
         a = "Likelihood A about reward:\n" + str(self.A)

         return pres + a

    def reset(self, A):

        self.visited_hint = False if self.agent.hint < 0 else True
        p = self.agent.p
        self.A = A

        # if self.visited_hint:
        #     self.p_obs = softmax([[0,3,-3,0],[0,-3,3,0]])[self.agent.hint-1]
        #
        # else:
        #     self.p_obs = softmax([[0,3,-3,0],[0,-3,3,0]])


########## Active Inference ######################


### Expected free energy ###

    def expected_free_energy(self, expected_state, expected_outcome, info=False):

        """
        Using the current expected state, calculate the expected free energy if we
        were to take a specific action. The expected free energy is the sum of
        - cost: minimizing it favors observations the agents prefers
        - ambiguity: quantifies the uncertainty of the mapping between state and
        observation. The higher, the more uncertain.

        expected_state: (2d-array)  The state the agent expects to be in at a
                                    particular time step t. Formally: q(s_t | policy)
        expected_outcome: (int)
        """

        # Complexity (= cost) term
        # Kullback-Leibler-Divergence between expected observation if action a is
        # taken and the agent's preferences in observations.
        # The higher the difference the less attractive the action to the agent.

        #cost = mean_square_error(expected_outcome, self.p_obs)
        cost = kl([expected_outcome, 1-expected_outcome], self.p_obs)

        # Ambiguity (or uncertainty) term
        # The lower the uncertainty, the more attractive the action to the agent.
        ambiguity = np.sum(expected_state * H(self.A))

        exp_free_energy = cost + ambiguity

        if info:
            print("Preferred observation:", self.p_obs)
            print("Expected outcome:", expected_outcome)

            print("Cost:", cost)
            print("Ambiguity:", ambiguity)
            print("Expected free energy:", exp_free_energy)
            print()

        return exp_free_energy


    def exp_free_energy_per_policy(self, q_state, q_outcome, info=False):

        """ Compute free energy of each policy.

        Returns: q_states_policy  (2d-array)    The expected future states if
                                                policy is followed
        """
        t = self.agent.time_step


        fe_policy = 0

        q_state = np.tile(q_state, 2).reshape(3,2,4)[t+1:]
        #print(q_state)
        q_outcome = q_outcome[t+1:]
        #print(q_outcome)

        for t, (q_s, q_o) in enumerate(zip(q_state, q_outcome)):

            if q_s.ndim == 2:
                action = np.argmax(q_s[0])
            else:
                action = np.argmax(q_s)


            if action == 3 and not self.visited_hint and t == 0:

                self.visited_hint = True
                self.update_A()

            if self.visited_hint:
                q_s = q_s[0]

            exp_free_energy = self.expected_free_energy(q_s, q_o)
            fe_policy += exp_free_energy

            if info:
                print("Action:", action)
                print("Expected free energy:", exp_free_energy)

        self.reset(self.agent.A)

        return fe_policy

    def exp_free_energy_all_policies(self, q_states, q_outcomes, info=False):

        """
        Computes the free energy of different policies. The free energy of each
        action is predicted and then summed up to obtain the policy's free energy.

        """
        exp_free_energies = list()

        for q_s, q_o in zip(q_states, q_outcomes):

            fe_policy = self.exp_free_energy_per_policy(q_s, q_o)
            exp_free_energies.append(fe_policy)


        exp_free_energies = np.array(exp_free_energies)

        if info:
            print("\n\nPolicy:\n", policies)
            print("\nFree energy:", exp_free_energies)

        return exp_free_energies

#### Sampling from the generative model ########


    def sample_states(self, info=False):

        """
        Computes the free energy of different policies. The free energy of each
        action is predicted and then summed up to obtain the policy's free energy.

        """

        q_state = list()
        policies = self.policies#[:, self.agent.time_step:]

        for policy in policies:

            q_states_policy = self.sample_states_per_policy(policy)
            q_state.append(q_states_policy)


        if info:
            print("\n\nPolicy:\n", policies)
            print("Q-states\n", np.array(q_state))


        return np.array(q_state)


    def sample_states_per_policy(self, policy, info=False):

        """ Compute free energy of each policy.

        Returns: q_states_policy  (2d-array)    The expected future states if
                                                policy is followed
        """

        current_state = np.average(self.agent.start_state, axis=0)
        q_states_policy = [current_state]

        for time_step, action in enumerate(policy):

            exp_state_after_action = current_state @ self.B[action]
            current_state = exp_state_after_action

            q_states_policy.append(exp_state_after_action)

            if info:
                print("Action:", action)
                print("Expected state after action", exp_state_after_action,'\n\n')

        return np.array(q_states_policy)


    def sample_outcomes(self, q_states):

        A = self.A
        outcomes = list()

        for q_state in q_states:

            outcome = list()

            for q_s in q_state:

                o = q_s @ A
                outcome.append(o)

                # Location 1 and 2 are absorbing states:
                # the agent does not continue to move or to receive feedback
                # after getting (only works here because state is perfectly known)
                action = np.argmax(q_s)

                if action == 1 or action == 2:
                    A = np.zeros(4)

            outcomes.append(outcome)
            A = self.A

        return np.array(outcomes)


###### Update belief when cue is seen ###

    def update_A(self):

        A = self.agent.belief_updating(1)
        #self.A = np.array([0, p, (1-p), 0])
        self.A = A


##### Computing the probability of action through bayesian model averaging ###

    def bayesian_averaging(self, free_energies, info=False):


        q_pi = softmax(-free_energies-np.amax(-free_energies), self.agent.temperature)
        self.q_pi = q_pi

        policies = self.policies[:, self.agent.time_step].flatten()
        action_prob = np.zeros(4)

        for i, action in enumerate(policies):
            action_prob[action] += q_pi[i]

        action_prob /= np.sum(action_prob)

        if info:
            print("Policy probability q(pi):\n", q_pi, '\n')
            print("Action probability q(u):", action_prob)

        return action_prob


### Expected reward (for rendering) #########
    def expected_reward(self, q_states, action):

        current_reward = 0
        discount = 1

        q_state_average = np.average(q_states, weights=self.q_pi, axis=0)

        for q in q_state_average[self.agent.time_step+1:]:

            current_reward += (q @ self.A) * discount
            discount /= 2

        return current_reward


######################## Learning ##########################

    def variational_free_energy(self, policy, q_state, true_reward):

        t = self.agent.time_step
        q_state = remove_zeros(q_state)

        a = np.log( remove_zeros( self.A )) * true_reward
        b = np.log( remove_zeros( self.B[policy[t-1]])) @ q_state[t-1]

        F = q_state[t] @ (np.log(q_state[t]) - b - a)

        return F

    def variational_fe_gradient(self, policy, q_state, true_reward, A):

        t = self.agent.time_step

        q_state = remove_zeros(q_state)
        q_state = np.vstack([q_state, q_state[-1]])
        policy = np.append(policy, policy[-1])

        a = np.log( remove_zeros( A ))  * true_reward
        b = np.log( remove_zeros( self.B[policy[t-1]])) @ q_state[t-1]
        c = np.log( remove_zeros( self.B[policy[t]])) @ q_state[t+1]

        gradient_F = (a + b + c) - np.log(q_state[t])

        return gradient_F




    def compute_new_A(self, F, policy, q_state, true_reward):

        t = self.agent.time_step
        t = 1

        a = np.log( remove_zeros(true_reward ))
        b = np.log( remove_zeros( self.B[policy[t-1]])) @ q_state[t-1]
        c = q_state[t] * (np.log(remove_zeros(q_state[t])) - b - a)

        A = q_state[t] * (np.log(remove_zeros(q_state[t])) - b - a) - F
        #print(A, softmax(-A, T=10))

        # Should be an exponential, insteadof softmax function:
        # see Sajid, 2019, page 14
        A = softmax(-A, T=self.agent.temperature)

        return A


    def gradient_descent(self, F, policy, q_state, true_reward, learn_rate=3, n_iter=5):

        a = self.A[1:3]
        A = self.A

        for _ in range(n_iter):

            diff = -learn_rate * self.variational_fe_gradient(policy, q_state, true_reward, A)
            F += diff

            A = self.compute_new_A(F, policy, q_state, true_reward)

        # A = np.zeros(4)
        # A[1:3] = a

        return A







if __name__ == "__main__":

    from Agent import Agent
    from Environment import Environment

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    env = Environment()
    a = Agent(env, food_is_left_prior=.5)
    lt = a.long_term

    ############ SAND BOX ##############

    a.time_step = 0

    q = lt.sample_states()
    o = lt.sample_outcomes(q)
    fe = lt.exp_free_energy_all_policies(q, o)

    print(fe)

    print(lt.bayesian_averaging(fe))
