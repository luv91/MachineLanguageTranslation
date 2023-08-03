# this works
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

# Path to the directory containing all your TensorBoard log directories
root_logdir = 'new runs 0.1 dropout'

# Dictionary to store DataFrames for each log directory
dataframes = {}

# Iterate through each subdirectory (each representing an experiment or run)
for logdir in os.listdir(root_logdir):
    logpath = os.path.join(root_logdir, logdir)
    ea = event_accumulator.EventAccumulator(logpath)
    ea.Reload()

    # Check if scalars tag is empty
    scalar_tags = ea.Tags().get('scalars', [])
    if not scalar_tags:
        print(f"No scalar data found in {logdir}. Skipping...")
        continue

    # Determine the maximum number of steps
    max_steps = max(len(ea.Scalars(tag)) for tag in scalar_tags)

    # Dictionary to store the data for this run
    run_data = {'step': list(range(max_steps))}

    # Iterate through all the scalar tags
    for tag in scalar_tags:
        # Retrieve the scalars for this tag
        scalars = ea.Scalars(tag)

        # Initialize the tag's data list
        run_data[tag] = []

        # Save the scalar values into the run_data dictionary, along with the steps
        for scalar in scalars:
            run_data[tag].append(scalar.value)

        # Pad the list with NaN values if it's shorter than max_steps
        run_data[tag] += [np.nan] * (max_steps - len(run_data[tag]))

    # Convert this run's data into a DataFrame
    run_df = pd.DataFrame.from_dict(run_data)

    # Save the DataFrame in the dataframes dictionary
    dataframes[logdir] = run_df

    # Save the DataFrame to a CSV file, including the logdir in the filename
    csv_filename = f'scalars_{logdir}.csv'
    run_df.to_csv(csv_filename, index=False)

# Print the log directory names and corresponding DataFrame names
for logdir, df in dataframes.items():
    print(f"logdir: {logdir}, DataFrame shape: {df.shape}")