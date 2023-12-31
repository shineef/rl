### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
# Modified By Yanhua Li on 09/09/2022 for gym==0.25.2
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    while True:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            new_v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, _ in P[s][a]:
                    new_v += action_prob * prob * (reward + gamma * value_function[next_state])
            value_function[s] = new_v
            delta = max(delta, abs(v - new_v))
        if delta < tol:
            break
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    for state in range(nS):
        q_values = np.zeros(nA)
        for action in range(nA):
            p = P[state][action]
            reward = sum([p_i[0] * p_i[2] for p_i in p])
            expected_next_value = sum([p_i[0] * value_from_policy[p_i[1]] for p_i in p])
            Q_value = reward + gamma * expected_next_value
            q_values[action] = Q_value
        best_action = np.argmax(q_values)
        new_policy[state] = np.zeros(nA)
        new_policy[state][best_action] = 1.0

    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
	############################
	# YOUR IMPLEMENTATION HERE #
    #                          #
	############################
    while True:
        V = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, V, gamma)
        if np.all(new_policy == policy):
            break
        policy = new_policy.copy()

    return new_policy, V


def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    V_new = V.copy()
    policy_new = np.zeros([nS, nA])
    ############################
    # YOUR IMPLEMENTATION HERE #
    #                          #
    ############################
    while True:
        delta = 0
        for s in range(nS):
            v = V_new[s]
            max_action_value = -np.inf
            for a in range(nA):
                action_value = sum(
                    [prob * (reward + gamma * V_new[next_state]) for prob, next_state, reward, _ in P[s][a]])
                if action_value > max_action_value:
                    max_action_value = action_value
            V_new[s] = max_action_value
            delta = max(delta, abs(v - V_new[s]))
        if delta < tol:
            break

    for s in range(nS):
        best_action = None
        max_action_value = -np.inf
        for a in range(nA):
            action_value = sum([prob * (reward + gamma * V_new[next_state]) for prob, next_state, reward, _ in P[s][a]])
            if action_value > max_action_value:
                max_action_value = action_value
                best_action = a
        policy_new[s][best_action] = 1.0

    return policy_new, V_new


def render_single(env, policy, render=False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game.
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob, _ = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            ############################
            # YOUR IMPLEMENTATION HERE #
            #                          #
            ############################
            action_probabilities = policy[ob]
            action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
            next_state, reward, done, _, info = env.step(action)
            total_rewards += reward
            ob = next_state
    return total_rewards





