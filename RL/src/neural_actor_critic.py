import numpy as np

# Policy initialization
def initialize_policy(n_actions):
    """
    Initialize a uniform policy over the given number of actions.

    Parameters:
        n_actions (int): The number of available actions.

    Returns:
        np.ndarray: A uniform probability distribution over actions.
    """
    return np.full(n_actions, 1 / n_actions)

# Policy update using Mirror Descent with Entropy Regularization
def mirror_descent_with_entropy(policy, rewards, eta, tau):
    """
    Update the policy using Mirror Descent with Entropy Regularization.

    Parameters:
        policy (np.ndarray): The current policy distribution.
        rewards (np.ndarray): The rewards associated with each action.
        eta (float): The learning rate.
        tau (float): The entropy regularization coefficient.

    Returns:
        np.ndarray: The updated policy distribution.
    """
    # Scale rewards and adjust the policy
    scaled_rewards = eta * rewards
    updated_policy = policy ** (1 - tau) * np.exp(scaled_rewards)
    return updated_policy / np.sum(updated_policy)  # Normalize to maintain a valid probability distribution

# Credit assignment for exploratory and exploitative values
def credit_assignment(reward, action, Q_E, Q_I, alpha, wI):
    """
    Update exploratory (Q_E) and exploitative (Q_I) value estimates.

    Parameters:
        reward (float): The received reward.
        action (int): The action taken.
        Q_E (np.ndarray): The exploratory value estimates for each action.
        Q_I (np.ndarray): The exploitative value estimates for each action.
        alpha (float): The learning rate.
        wI (float): The weight of influence of Q_I on Q_E.

    Returns:
        tuple: Updated (Q_E, Q_I) value estimates.
    """
    Q_E[action] = Q_E[action] + alpha * (- Q_E[action] + reward - wI/alpha * Q_I[action])
    Q_I[action] = Q_I[action] + alpha * (- Q_I[action] + reward - Q_E[action])
    return Q_E, Q_I
