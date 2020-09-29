import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

###
##TO DO
## Try adding q states information to exp_free_energy

np.seterr(invalid='ignore', divide='ignore')

## Based on https://medium.com/@solopchuk/tutorial-on-active-inference-30edcf50f5dc
## Tutorial on Active Inference

##### Helper functions ############

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def KL(a, b):
    # Kullback-Leibler divergence
    # From https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def H(p):
    # Ambiguity
    p = np.array(p)
    H = -np.sum(np.where(p != 0, p * np.log(p), 0))

    return H

def mutual_information(p, a, b):
    pass

######## Helper Variables ###########################

unique_actions = [0, 1]
unique_states = [0, 1]

# precision = np.zeros(policies.shape[0])

######## Generative model #############################

# The generative model p(o, s)

# Index 0: probability of EITHER observation "hungry"
#           OR true hidden state "hungry"
#           OR action "do nothing"

# Index 1: probability of EITHER observation "full"
#           OR true hidden state "full"
#           OR action "eat"

# Prior preferences on observations
# p(o)

p_obs = [1, 0]


# Prior belief on hidden states
# p(s)
# Not used

p_states = [0.5, 0.5]

# Likelihood p(observation | state)
# A[y, x]
# Y-axis: state
# X-axis: observation

A = np.array([[0.9, 0.1], [0.3, 0.7]])
A = np.array([[1, 0], [0, 1]])

# Transition probability p(state_t | state_t-1, action)
# B[x, y, z]
# X-axis: Current state (at t)
# Y-axis: action
# Z-axis: Next state (at t+1)

B = np.array([[[0.8, 0.2], [0.3, 0.7]],[[0.6, 0.4], [0.15, 0.85]]])
B = np.array([[[1, 0], [0, 1]],[[1, 0], [0, 1]]])

######### Prediction ################################


def expected_free_energy(q_state, action, p_obs=p_obs):

    """
    Using the current expected state, calculate the expected free energy if we
    were take a specific action. The expected free energy is the sum of
    - cost: minimizing it favors observations the agents prefers
    - ambiguity: quantifies the uncertainty of the mapping between state and
    observation

    expected_state: (2d-array)  The state the agent expects to be in at a
                                particular time step t-1. Formally: q(s_t | policy)
    p_obs:          (2d-array)  The observation the agent prefers.
                                Formally: p(observation)
    action: (int)               Action taken at time step t-1.

    ### Questions ###
    How does this prediction work?
    How does Friston predicts free energy in his papers?
    At which time step is the action done? t or t-1?
    """
    # Complexity term
    # Kullback-Leibler-Divergence between expected observation if action a is
    # taken and the agent's preferences in observations
    q_state = np.array(q_state)[-1]
    #print("Q\n", q_state)
    #exp_current_state = q_state[-1]
    exp_next_state = B[:, action] @ q_state
    expected_obs = A @ exp_next_state

    cost =  KL(expected_obs, p_obs)

    print(exp_next_state)

    # Ambiguity term
    ambiguity = 0

    for s in unique_states:
        ambiguity += exp_next_state[s] * H(A[s])
        #ambiguity += q_state * H(A[s])

    #print("cost", cost, "ambiguity", ambiguity, "expected_obs", expected_obs)

    exp_free_energy = cost + ambiguity
    print("Action", action, "\tExpected free energy:", exp_free_energy)

    return exp_free_energy


def exp_free_energy_per_policy(policies, start_state, p_obs=p_obs):
    """
    Computes the free energy of different policies. The free energy of each
    action is predicted and then summed up to obtain the policy's free energy.

    policies:    (2d-array) The different policies the agent considers.
    start_state: (1d-array)
    p_obs:          (1d-array)  The observation the agent prefers.
                                Formally: p(observation)
    """

    q_state = []
    exp_free_energies = []
    start_state = np.array(start_state)#.reshape(1, 2)

    for policy in policies:

        print("Policy:", policy)

        fe_policy = 0
        current_state = start_state

        ##
        q_states_policy = []
        q_states_policy.append(current_state)
        print("Start state", start_state)

        ## Compute free energy of each policy
        for time_step, action in enumerate(policy):

            exp_free_energy = expected_free_energy(q_states_policy, action, p_obs)
            fe_policy += exp_free_energy

            exp_state_after_action = (current_state @ B[:, action]).squeeze()

            q_states_policy.append(exp_state_after_action)
            current_state = exp_state_after_action

        print("Free energy:", fe_policy, "\n")
        q_state.append(q_states_policy)
        exp_free_energies.append(fe_policy)

    q_state = np.array(q_state).squeeze()
    exp_free_energies = np.array(exp_free_energies)

    # Precision determines the exploration vs. exploitation trade-off and favors
    # some policies over others.
    # exp_free_energies *= precision

    return exp_free_energies, q_state

## Transform it to probability of of policies

def policy_probability(free_energies):
    """
    Translates free energies of policies into the approximate posterior on policies
    q(policy). The less free energy was predicted for a policy, the higher the
    probability it will be used.

    free_energies: (1d-array) The free energy of each policy. The index indicates
                              to which policy the free energy belongs to.
    """
    return softmax(-free_energies)


## Bayesian average over policies

def bayesian_averaging(free_energies, q_state_per_policy):
    """


    free_energies: (1d-array) The free energy of each policy. The index indicates
                              to which policy the free energy belongs to.
    """
    policy_prob = policy_probability(free_energies)
    repeats = q_state_per_policy.shape[1] * q_state_per_policy.shape[2]
    policy_prob = np.repeat(policy_prob, repeats).reshape(q_state_per_policy.shape)

    q_state = policy_prob * q_state_per_policy
    q_state = q_state.sum(axis=0)

    return q_state


############# Act ######################

## Act such as we observe what we expect

def action_selection(q_state, current_state, time_step):
    """

    q_state: (2d-array)
    time_step: (int)     Time step t.
    """

    ## Expected observations, multiply
    ## - belief on the hidden state at the next time step q(s_t+1)
    ## - state-observation mapping p(o|s) ('A' matrix)

    expected_observation = q_state[time_step+1] @ A
    print("Expected obs", expected_observation)
    ## Expected observations if we were to take a certain action: multiply
    ## - Belief on the hidden state at the current time step q(s_t)
    ## - Action-specific state-transition matrix B (to get the hypothetical next state)
    ## - State-observation matrix A (to get the hypothetical observation)

    exp_observation_after_action = list()

    for a in unique_actions:
        #print(current_state, current_state[time_step])
        exp_observation_after_action.append(current_state @ B[:, a] @ A)

    exp_observation_after_action = np.array(exp_observation_after_action)
    #print("Expected obs after action\n", exp_observation_after_action)

    # Pick action minimizing KL divergence of what we expect to see
    # in the next time step, and what we were to see if we picked a specific action.

    free_energy = list()

    for a in unique_actions:

        fe_per_action = KL(exp_observation_after_action[a], expected_observation)
        free_energy.append(fe_per_action)

    free_energy = np.array(free_energy)
    print("FE", free_energy)
    chosen_action = np.argmin(free_energy)
    print("Take action", chosen_action, end='\n\n')

    return chosen_action


def execute_action(current_state, action):
    """

    current_obs: (1d-array) Current perception of the agent. Probability
                            distribution of perceiving observation 0
    action:      (int)

    ## TO FIX(?) ##
    - Should include true hidden environmental model instead of generative model
    (matrices A, B)
    - No need to compute observation
    """

    #
    # current_state = (current_obs * A).sum(axis=0)
    # print("Current state:", current_state)

    #state_after_action = (current_state * B[:, action].T).sum(axis=0)
    state_after_action = current_state @ B[:, action]
    print("Next state:", state_after_action)

    obs_after_action = (state_after_action.T * A).sum(axis=1)
    obs_after_action = softmax(obs_after_action)
    print("Next obs:", obs_after_action)

    return state_after_action, obs_after_action


def act(q_state, start_state):

    """
    q_state: (2d-array)
    start_obs: (1d-array)
    """

    #current_obs = start_obs
    current_state = start_state
    actions_taken = []

    for time_step in range(q_state.shape[0]-1):

        action = action_selection(q_state, current_state, time_step)
        actions_taken.append(action)
        current_state, current_obs = execute_action(current_state, action)

    return actions_taken


########### Predict and Act ##################

def predict_and_act(p_obs, start_state, policies):

    """
    p_obs:       (1d-array)  The observation the agent prefers.
                             Formally: p(observation)
    policies:    (2d-array) The different policies the agent considers.
    """

    print("## Prediction ##")
    fe, qs = exp_free_energy_per_policy(policies, start_state, p_obs)
    qs_final = bayesian_averaging(fe, qs)

    print("\n## Action ##")
    print("\nActions taken:", act(qs_final, start_state))
    print("Policy with lowest FE:\n", policies[np.where(fe == fe.min())], "\nFree energy:", fe.min())
    print("q(state|best policy):\n", qs[np.where(fe == fe.min())])
    #print("All policies\n", policies)
    #print("All free energies", fe)
    #print("q(state|policy)\n", qs)
    print("Final q(state)\n", qs_final)

################################################



qs = np.arange(24).reshape(4, 3, 2)
fe = np.arange(4)

#print(bayesian_averaging(fe, qs))

#q_state = np.array([[0, 1], [0.8, 0.2], [0.01, 0.99]])
start_state = [1, 0]
start_obs = [1, 0]

policies = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
policies = np.array([np.array(i) for i in itertools.product([0, 1], repeat=5)])

p_obs = np.array([[1, 0], [1, 0], [0, 1], [1, 0]])
p_obs = np.broadcast_to(np.array([0.5, 0.5]), (policies.shape[1], 2))
p_obs = np.array([0.6, 0.4])

predict_and_act(p_obs, start_state, policies)

#print(exp_free_energy_per_policy(policies, np.array([0.1, 0.9])))
#print(act(q_state, [0, 1]))
