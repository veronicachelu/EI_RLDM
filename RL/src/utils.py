import numpy as np

# Function to compute the entropy of a policy
def compute_entropy(policy):
    """
    Compute the entropy of a probability distribution (policy).

    Parameters:
    - policy (array-like): A probability distribution array.

    Returns:
    - float: The entropy of the distribution.
    """
    epsilon = 1e-12  # Small constant to avoid log(0)
    return -np.sum(policy * np.log(policy + epsilon))

# Schedule utility functions
def exponential_decay(initial_value, decay_rate, time_step):
    """
    Compute the value of an exponentially decaying schedule.

    Parameters:
    - initial_value (float): The starting value of the schedule.
    - decay_rate (float): The rate at which the value decays.
    - time_step (int): The current time step.

    Returns:
    - float: The value at the current time step.
    """
    return initial_value * np.exp(-decay_rate * time_step)

def constant_schedule(value, _):
    """
    Return a constant value for a schedule.

    Parameters:
    - value (float): The constant value to return.
    - _ (int): Ignored, included for consistent function signature.

    Returns:
    - float: The constant value.
    """
    return value
