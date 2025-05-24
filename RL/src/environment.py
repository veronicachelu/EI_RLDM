import numpy as np

# Continual Bandit environment
class ContinualBandit:
    """
    A continual bandit environment where tasks change over time, requiring adaptation.

    Attributes:
        rng (np.random.Generator): Random number generator for reproducibility.
        n_actions (int): Number of actions available to the agent.
        intervals (list): List of intervals indicating task change steps.
        tasks (list): List of tasks, each represented as (reward means, reward probabilities).
        reward_prob (float): Probability of receiving a reward.
        reward_var (float): Variance of the reward noise.
    """

    def __init__(self, rng, n_actions, intervals, tasks, reward_prob=1.0, reward_var=0.5):
        self.rng = rng
        self.n_actions = n_actions
        self.intervals = intervals
        self.tasks = tasks
        self.reward_prob = reward_prob
        self.reward_var = reward_var
        self.change_steps = np.cumsum(intervals)
        self.task_id = 0
        self.current_task = self.tasks[self.task_id]
        self.current_task_means = self.current_task[0]
        self.current_task_probs = self.current_task[1]

    def step(self, action, step):
        """
        Simulates taking an action at a given step and returns the reward.

        Parameters:
            action (int): The action taken by the agent.
            step (int): The current time step.

        Returns:
            float: The reward received.
        """
        # Update the task if the current step is a task change step
        if step in self.change_steps:
            self.task_id += 1
            self.current_task = self.tasks[self.task_id]
            self.current_task_means = self.current_task[0]
            self.current_task_probs = self.current_task[1]

        # Compute task-specific rewards
        task_rewards = [
            self.rng.choice(self.current_task_means[0], p=[self.current_task_probs[0], 1 - self.current_task_probs[0]]),
            self.rng.choice(self.current_task_means[1], p=[self.current_task_probs[1], 1 - self.current_task_probs[1]])
        ]
        task_reward = task_rewards[action]

        # Add noise to the reward
        reward = self.rng.normal(
            self.rng.choice([0, task_reward], p=[1 - self.reward_prob, self.reward_prob]),
            self.reward_var
        )
        return reward

    def expected_rewards(self):
        """
        Computes the expected rewards for each action in the current task.

        Returns:
            list: A list of expected rewards for each action.
        """
        r = [
            self.current_task_means[i][0] * self.current_task_probs[i] +
            self.current_task_means[i][1] * (1 - self.current_task_probs[i])
            for i in range(2)
        ]
        return [r[i] * self.reward_prob for i in range(2)]


def generate_task_distribution(
        task_dist, n_tasks, init_reward_means=([1, -1], [-1, 1]), init_reward_probs=(1, 1), task_reward_probs=None
):
    """
    Generates a distribution of tasks for the continual bandit environment.

    Parameters:
        task_dist (str): Type of task distribution ("DetPRL" or "PRL").
        n_tasks (int): Number of tasks to generate.
        init_reward_means (list): Initial reward means for tasks.
        init_reward_probs (tuple): Initial reward probabilities for tasks.
        task_reward_probs (list): Reward probabilities for each task (only for "PRL").

    Returns:
        list: A list of tasks represented as (reward means, reward probabilities).
    """
    current_task_means = init_reward_means
    current_task_probs = init_reward_probs
    current_task = (current_task_means, current_task_probs)

    task_list = []

    if task_dist == "DetPRL":
        for i in range(n_tasks):
            if i % 2 == 0:
                task_list.append(current_task)
            else:
                inverted_probs = (1 - current_task_probs[0], 1 - current_task_probs[1])
                task_list.append((current_task_means, inverted_probs))

    elif task_dist == "PRL":
        task_list.append(current_task)
        for i, reward_probs in enumerate(task_reward_probs):
            task_list.append((current_task_means, reward_probs[i]))

    return task_list
