import numpy as np
import os

# Set parameters
num_train_trials = 120000  # Total number of trials
num_test_trials = 100  # Total number of trials
num_valid_trials = 100  # Total number of trials
num_trials = num_train_trials + num_valid_trials + num_test_trials
time_steps = 60  # Number of time steps in a trial
num_inputs = 2  # Two input streams representing sensory evidence
num_outputs = 2  # Binary decision output
stimulus_magnitude = 3.2  # Fixed stimulus magnitude
coherence_range = (-0.2, 0.2)  # Range of stimulus difficulties
stimulus_onset = 10  # Stimulus starts at step 10
stimulus_duration = 21  # Stimulus duration
noise_level = 0.05  # Input noise level

# Initialize arrays
inputs_series = np.zeros((num_trials, time_steps, num_inputs))
targets_series = np.zeros((num_trials, time_steps, num_outputs))
mask_series = np.ones((num_trials, time_steps))
coherence_series = np.zeros((num_trials))

# Generate trials
# coherence_list = np.linspace(*coherence_range, num_train_trials)
for trial in range(num_train_trials):
    if np.random.rand() < 1/2:
        coherence = 0
    else:
        coherence = np.random.uniform(*coherence_range)
    noise = np.random.normal(0, noise_level, (time_steps, num_inputs))

    # Generate input streams
    inputs = np.full((time_steps, num_inputs), 0.2)  # Baseline input
    inputs[stimulus_onset:stimulus_onset + stimulus_duration, 0] += (
            1 + coherence*stimulus_magnitude
    )
    inputs[stimulus_onset:stimulus_onset + stimulus_duration, 1] += (
            1 - coherence*stimulus_magnitude
    )
    inputs += noise  # Add Gaussian noise

    # Determine decision outcome
    targets = np.full((time_steps, num_outputs), 0.2)
    decision_threshold = 0.25
    if coherence != 0:
        decision = np.argmax(inputs[stimulus_onset + stimulus_duration - 1])
        # targets[stimulus_onset:stimulus_onset + stimulus_duration, decision] = 1
        targets[stimulus_onset:, decision] = 1
        mask_series[trial, stimulus_onset:stimulus_onset + stimulus_duration] = 0  # Mask stimulus period

        # if np.abs(inputs[0, stimulus_onset + stimulus_duration - 1] -
        #           inputs[1, stimulus_onset + stimulus_duration - 1]) < decision_threshold:
        #     decision = np.random.choice([0, 1])  # Assign random choice if undecided

    # Store data
    inputs_series[trial] = inputs
    targets_series[trial] = targets
    coherence_series[trial] = coherence
    
coherence_list = np.linspace(*coherence_range, num_valid_trials)
for j, trial in enumerate(range(num_train_trials, num_train_trials + num_valid_trials)):
    coherence = coherence_list[j]
    noise = np.random.normal(0, noise_level, (time_steps, num_inputs))

    # Generate input streams
    inputs = np.full((time_steps, num_inputs), 0.2)  # Baseline input
    inputs[stimulus_onset:stimulus_onset + stimulus_duration, 0] += (
            1 + coherence*stimulus_magnitude
    )
    inputs[stimulus_onset:stimulus_onset + stimulus_duration, 1] += (
            1 - coherence*stimulus_magnitude
    )
    inputs += noise  # Add Gaussian noise

    # Determine decision outcome
    targets = np.full((time_steps, num_outputs), 0.2)
    decision = np.argmax(inputs[stimulus_onset + stimulus_duration - 1])
    targets[stimulus_onset:, decision] = 1

    # Store data
    inputs_series[trial] = inputs
    targets_series[trial] = targets
    coherence_series[trial] = coherence
    mask_series[trial, stimulus_onset:stimulus_onset + stimulus_duration] = 0  # Mask stimulus period

coherence_list = np.linspace(*coherence_range, num_test_trials)
for j, trial in enumerate(range(num_train_trials + num_valid_trials,
                            num_train_trials + num_valid_trials + num_test_trials)):
    coherence = coherence_list[j]
    noise = np.random.normal(0, noise_level, (time_steps, num_inputs))

    # Generate input streams
    inputs = np.full((time_steps, num_inputs), 0.2)  # Baseline input
    inputs[stimulus_onset:stimulus_onset + stimulus_duration, 0] += (
            1 + coherence*stimulus_magnitude
    )
    inputs[stimulus_onset:stimulus_onset + stimulus_duration, 1] += (
            1 - coherence*stimulus_magnitude
    )
    inputs += noise  # Add Gaussian noise

    # Determine decision outcome
    targets = np.full((time_steps, num_outputs), 0.2)
    decision = np.argmax(inputs[stimulus_onset + stimulus_duration - 1])
    targets[stimulus_onset:, decision] = 1

    # Store data
    inputs_series[trial] = inputs
    targets_series[trial] = targets
    coherence_series[trial] = coherence
    mask_series[trial, stimulus_onset:stimulus_onset + stimulus_duration] = 0  # Mask stimulus period

# Save dataset
dataset_path = "~/logs/EI_RLDM/decision_making/datasets/"
os.makedirs(dataset_path, exist_ok=True)
np.savez(
    f"{dataset_path}/decision_making_data.npz",
    inputs_series=inputs_series,
    targets_series=targets_series,
    coherence_series=coherence_series,
    mask_series=mask_series
)

print("Dataset saved successfully.")