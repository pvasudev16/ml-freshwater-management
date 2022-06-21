import pandas as pd
import numpy as np
import datetime
from tensorflow import keras
from keras.models import Sequential
from keras import Input
from keras.layers import Bidirectional, LSTM, Dense, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt

# Read pre-processed output
max_track_length = 920
output = pd.read_csv("./processed/imputed_sentinel_a_b_data.csv", parse_dates=["date"])

df_train = output.iloc[0:200*920]
df_test = output.iloc[201*920::]

def prepare_data(datain, timestep):
    # Get the lake water levels as numpy arrays
    lake_water_levels = np.array(datain["lake_water_level"])
    in_situ_lake_water_levels = np.array(datain["in_situ_lake_water_level"])
    
    # Get the number of unique dates. For e.g. in our training data, it's 200.
    number_of_dates = len(pd.unique(datain["date"]))
    
    # Number of windows we can fit into the data
    number_of_windows = number_of_dates - (2 * timestep) + 1
    
    # Sliding window across the data
    for d in range(0, number_of_dates - (2 * timestep) + 1):
        X_start = d * max_track_length # Starting index
        X_end = (d + timestep) * max_track_length # Finishing index
        
        Y = (d + np.arange(timestep)) * max_track_length # Indices for getting in-situ data
        
        if d==0:
            X_comb = lake_water_levels[X_start:X_end]
            Y_comb = in_situ_lake_water_levels[Y]
        else:
            X_comb = np.append(X_comb, lake_water_levels[X_start:X_end])
            Y_comb = np.append(Y_comb, in_situ_lake_water_levels[Y])

    # Reshape input and target arrays
    X_out = np.reshape(X_comb, (number_of_windows, timestep*max_track_length, 1))
    Y_out = np.reshape(Y_comb, (number_of_windows, timestep, 1))
    return X_out, Y_out

X_train, Y_train = prepare_data(datain=df_train, timestep=5)
X_test, Y_test = prepare_data(datain=df_test, timestep=5)

# Define LSTM
model = Sequential(name="LSTM-Model")
model.add(
    Input(
        shape=(X_train.shape[1], X_train.shape[2]),
        name="Input-Layer"
    )
)
model.add(
    Bidirectional(
        LSTM(
            units=32,
            activation="tanh",
            recurrent_activation="sigmoid",
            stateful=False,
        ),
        name="Hidden-LSTM-Encoder-Layer"
    )
)
model.add(
    RepeatVector(
        Y_train.shape[1],
        name="Repeat-Vector-Layer"
    )
)
model.add(
    Bidirectional(
        LSTM(
            units=32,
            activation="tanh",
            recurrent_activation="sigmoid",
            stateful=False,
            return_sequences=True
        ),
        name="Hidden-LSTM-Decoder-Layer"
    )
)
model.add(
    TimeDistributed(
        Dense(
            units=1,
            activation="linear"
        ),
        name="Output-Layer"
    )
)

# Compile model
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["MeanSquaredError", "MeanAbsoluteError"],
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None
)

# Fit the model
history = model.fit(
    X_train,
    Y_train,
    batch_size=1,
    epochs=1000,
    verbose=1,
    callbacks=None,
    validation_split=0.2,
    # validation_data=(X_test, Y_test)
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=100,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=True
)

model.save("My_LSTM")

# Make predictions on the training data
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# Print Performance Summary
print("")
print('-------------------- Model Summary --------------------')
model.summary() # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
print("Too many parameters to print but you can use the code provided if needed")
print("")

print('-------------------- Evaluation on Training Data --------------------')
for item in history.history:
    print("Final", item, ":", history.history[item][-1])
print("")

# Evaluate the model on the test data using "evaluate"
print('-------------------- Evaluation on Test Data --------------------')
results = model.evaluate(X_test, Y_test)
print("")

print('-------------------- Test Data Predictions --------------------')
print(pred_test)