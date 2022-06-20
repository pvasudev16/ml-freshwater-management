import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import matplotlib
# import cartopy.crs as ccrs # Projections
# import cartopy.feature as cfeature
# import cartopy
import datetime



sentinel_data_A = pd.read_csv("data/Sentinel_3A_water_level_Version0.csv")
sentinel_data_A = sentinel_data_A.rename(
    columns={
        "Date (YYYYMMDD)" : "date",
        "Lake_name" : "lake_name",
        "Latitude" : "latitude",
        "Longitude" : "longitude",
        "Relaive_orbit" : "relative_orbit",
        "Lake water level (m)" : "lake_water_level"
    }
)


lake_winnipeg = sentinel_data_A[
    sentinel_data_A["lake_name"] == "Winnipeg"
]

# Add ADM error
def add_adm_error(lake_winnipeg):
    lake_winnipeg["error"] = 0
    unique_dates = pd.unique(lake_winnipeg["date"])
    for date in unique_dates:
        median_water_level = lake_winnipeg.loc[
            lake_winnipeg["date"] == date,
            "lake_water_level"
        ].median()
        lake_winnipeg.loc[
            lake_winnipeg["date"] == date,
            "error"
        ] = lake_winnipeg.loc[
            lake_winnipeg["date"] == date,
            "lake_water_level"
        ].apply(lambda x: np.abs(x - median_water_level) + 0.25)
    return lake_winnipeg

def remove_adm_error(lake_winnipeg):
    lake_winnipeg = lake_winnipeg.drop(columns=["error"])
    return lake_winnipeg

lake_winnipeg = add_adm_error(lake_winnipeg)


# Reject outliers that are more than 2 std dev from the mean.
lake_winnipeg["mean_track_lake_water_level"] = lake_winnipeg.groupby("date")["lake_water_level"].transform("mean")
lake_winnipeg["std_track_lake_water_level"] = lake_winnipeg.groupby("date")["lake_water_level"].transform("std")
number_of_standard_deviations = 2

lake_winnipeg["reject"] = lake_winnipeg.apply(
    lambda row: True if np.abs(
        row["lake_water_level"] - row["mean_track_lake_water_level"]
    ) > number_of_standard_deviations*row["std_track_lake_water_level"] else False,
    axis=1
)

class KalmanFilter:
    def __init__(self):
        self.m_t = 0 # number of points in this time step
        self.x_t_prior = 0 # x_t prior, scalar
        self.x_t = 0 # x_t update, scalar
        self.F_t = 0 # transition matrix, scalar
        self.H_t = 0 # model matrix (m_t, 1)
        self.sigma_t_prior = 0 # model noise, prior; scalar
        self.S_t_prior = 0 # observation noise, prior; (m_t, m_t)
        self.sigma_t = 0 # model noise, update; scalar
        self.S_t = 0 # observation noise, update; (m_t, m_t)
        self.y_t_prior = 0 # predictions for observations based on x_t_prior
        self.y_t = 0 # observations; (m_t, 1)
        self.K_t = 0 # Kalman gain
        self.R_t = 0 # Covariance of noise in the observations at time t
        self.Q_t = 0 # Covariance of the model noise

        # KalmanNet features
        # F1: "innovation" difference between the lake water level data and y_t_prior
        self.innovation = 0

        # F2: "observation_difference": difference between y_t and y_{t-1}
        self.y_t_minus_one = 0
        self.observation_difference = 0

        # F3: "forward evolution difference": difference between x_t and x_{t-1}
        self.x_t_minus_one = 0
        self.forward_evolution_difference = 0
        
    def initialize(self, lake_data_0):
        # lake_data_0 is the lake data for the zeroth time step.
        # In our Lake Winnipeg case, it will be something like this;
        # lake_data_0 = lake_winnipeg.loc[
        #    lake_winnipeg["date"] = pd.unique(lake_winnipeg["date"])[0]
        #]
        
        # Initialize the x_t_prior to the lake water level that has the lowest
        # error estimate
        water_level_with_smallest_error = lake_data_0.loc[
            lake_data_0['error'].idxmin()
        ]['lake_water_level']
        self.x_t_prior = water_level_with_smallest_error
        
        # Initialize the model noise to unity
        self.sigma_t_prior = 1.
        
        # Define the variables from the data
        self.input_new_data(lake_data_0)
        
        # Set the transition "matrix"
        self.F_t = 1
        
        # Set the...?
        self.Q_t = 0.05**2
        
    def input_new_data(self, lake_data):
        # Set the number of points in this time step
        self.m_t = len(lake_data)
        
        # Set the model matrix
        self.H_t = np.ones((self.m_t, 1))
        
        # Set the observation noise matrix.
        self.R_t = np.zeros((self.m_t, self.m_t))
        np.fill_diagonal(self.R_t, lake_data["error"])
        self.S_t_prior = self.R_t + np.matmul(self.H_t, np.transpose(self.H_t)) * self.sigma_t_prior
        
        # Initialize the observations
        # Set the observations at time t to the the "t-1" ones and then add the newest data to y_t
        if isinstance(self.y_t_minus_one, int) or isinstance(self.y_t_minus_one, float):
            self.y_t_minus_one = np.array([self.y_t])
        else:
            self.y_t_minus_one = self.y_t
        self.y_t = np.array(lake_data["lake_water_level"]).reshape(self.m_t, 1)
        
        # Initialize the estimates based on the priors
        self.y_t_prior = self.H_t  * self.x_t_prior

        # Calculate F1: "innovation"
        self.innovation = self.get_innovation()

        # Calculate F2: "observation difference"
        self.observation_difference = self.get_observation_difference()


    
    def calculate_kalman_gain(
        self,
        sigma_t_prior, # model noise, prior
        S_t_prior, # Observation noise, prior
        H_t, # model matrix
    ):
        return np.matmul(
            sigma_t_prior * np.transpose(H_t),
            np.linalg.pinv(S_t_prior)
        )
        
    def update(self):
        self.K_t = self.calculate_kalman_gain(
            self.sigma_t_prior,
            self.S_t_prior,
            self.H_t
        )
        
        # Update state variable.
        # Store the previous iteration as t-1
        self.x_t_minus_one = self.x_t
        self.x_t = self.x_t_prior + np.matmul(
            self.K_t,
            self.y_t - self.y_t_prior
        ).item()

        # Calculate F3 forward evolution difference
        self.forward_evolution_difference = self.get_forward_evolution_difference()

        # Update sigma_t
        self.sigma_t = self.sigma_t_prior - np.matmul(
            self.K_t,
            np.matmul(
                self.S_t_prior,
                np.transpose(self.K_t)
            )
        ).item()
    
    def predict(self): # Is a better name predict_next_prior?
        self.x_t_prior = self.x_t
        self.sigma_t_prior = self.sigma_t + self.Q_t
    
    # Add functions to calculate features for KalmanNet
    # F1: "innovation"
    def get_innovation(self):
        return self.y_t - self.y_t_prior
    
    def get_observation_difference(self):
        length_of_y_t = len(self.y_t)
        length_of_y_t_minus_one = len(self.y_t_minus_one)

        if length_of_y_t == length_of_y_t_minus_one:
            return self.y_t - self.y_t_minus_one
        elif length_of_y_t > length_of_y_t_minus_one:
            padding_array = np.random.normal(
                loc=np.mean(self.y_t_minus_one),
                scale=np.std(self.y_t_minus_one),
                size=length_of_y_t - length_of_y_t_minus_one
            )
            return self.y_t - np.vstack((self.y_t_minus_one, padding_array.reshape((padding_array.shape[0], 1))))
        else:
            return self.y_t - self.y_t_minus_one[0:length_of_y_t]
    
    def get_forward_evolution_difference(self):
        return self.x_t - self.x_t_minus_one


times = pd.unique(lake_winnipeg["date"])
lake_water_levels = np.zeros(np.array(times).shape)
oHai = KalmanFilter()
number_of_time_points = len(times)
lake_winnipeg["innovation"] = 0.
lake_winnipeg["observation_difference"] = 0.
lake_winnipeg["forward_evolution_difference"] = 0.

for i, time in enumerate(times):
    # Initialize
    if i == 0:
        oHai.initialize(
            lake_winnipeg.loc[
                lake_winnipeg["date"] == time
            ]
        )

        # Write F1 innovation to data frame
        lake_winnipeg.loc[
            lake_winnipeg["date"]==time,
            "innovation"
        ] = oHai.innovation.reshape((len(oHai.innovation), 1))

        # Don't calculate F2 on the first iteration
    
    else:
        oHai.input_new_data(
            lake_winnipeg.loc[
                lake_winnipeg["date"] == time
            ]
        )

        # Write F1 innovation to data frame
        lake_winnipeg.loc[
            lake_winnipeg["date"]==time,
            "innovation"
        ] = oHai.innovation.reshape((len(oHai.innovation), 1))

        # Write F2 observation difference to data frame
        lake_winnipeg.loc[
            lake_winnipeg["date"]==time,
            "observation_difference"
        ] = oHai.observation_difference.reshape((len(oHai.observation_difference), 1))

        # Write F3 forward evolution difference to data frame
        lake_winnipeg.loc[
            lake_winnipeg["date"]==time,
            "forward_evolution_difference"
        ] = oHai.forward_evolution_difference
     
     # Update
    oHai.update()
    
    # Get the lake water levels
    lake_water_levels[i] = oHai.x_t # This should be a variable like y_k which is just x_k?
    
    oHai.predict()
    
    if (i%100 == 0) or (i == number_of_time_points - 1):
        percentage_complete = i/(number_of_time_points - 1) * 100.
        print("Processing %d, %0.02f%% complete"%(time, percentage_complete))

lake_winnipeg.to_csv("./processed_lake_water_data.csv")

# Form a baseline comparison
def rms(values):
    return np.sqrt(sum(values**2)/len(values))

baseline_results = lake_winnipeg[
    [
        "date",
        "lake_water_level",
        "error"
    ]
].groupby("date").agg(
    {
        "lake_water_level" : "median",
        "error": rms
    }
).reset_index()
baseline_results.loc[:, "date_as_datetime"] = pd.to_datetime(baseline_results.loc[:, "date"], format="%Y%m%d")
time_as_datetime = pd.to_datetime(times, format="%Y%m%d")


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(time_as_datetime, lake_water_levels, zorder=10, label="Kalman Filtered")
ax.plot(
    baseline_results["date_as_datetime"],
    baseline_results["lake_water_level"],
    'r--',
    linewidth=0.5,
    label="Median of each track"
)
ax.set_xlabel('Date', labelpad=20)
ax.set_ylabel('Lake Winnipeg water levels (m)')

# Set labels to be every six months
locator = matplotlib.dates.MonthLocator((1, 7))
fmt = matplotlib.dates.DateFormatter('%b-%Y')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(fmt)
for x in ax.get_xticklabels():
    x.set_rotation(45)

ax.legend()
plt.show()