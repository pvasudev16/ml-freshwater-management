import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import matplotlib
import cartopy.crs as ccrs # Projections
import cartopy.feature as cfeature
import cartopy
import datetime
import matplotlib.font_manager as fm

# This file generates the plots we used in our presentation. It is outlined as follows:
#    - Read in the Sentinel A data for Lake Winnipeg
#    - Figure 1: Sentinel A ground tracks over Lake Winnipeg
#    - Figure 2: Outlier rejection (reject points +/- 2*sigma from the mean)
#    - Figure 3: Mean lake water levels on each day
#    - Figure 4: Kalman filtered lake water levels; superimposed on median lake water levels
#    - Figure 5: Neural network prediced lake water levels; superimposed on median lake water levels and in-situ data
#                Uses both Sentinel A and Sentinel B data

# Constants for plotting
FIG_WIDTH_INCHES = 6.5
FIG_HEIGHT_INCHES = 6.5
TITLE_SIZE = 15
TICK_LABEL_SIZE = 14
LABEL_SIZE = 14
TEXT_COLOUR="w"

# Read Sentinel A Lake Winnipeg data
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

####
#### FIGURE 1: Sentinel A ground tracks over Lake Winnipeg
####
# Get the extent from the data
extent = [
    lake_winnipeg["longitude"].min(),
    lake_winnipeg["longitude"].max(),
    lake_winnipeg["latitude"].min(),
    lake_winnipeg["latitude"].max(),
]
central_lon = np.mean(extent[:2])
central_lat = np.mean(extent[2:])

# High resolution lakes
lakes_50m = cfeature.NaturalEarthFeature('physical', 'lakes', '10m')

# Land, river, and lakes
fig = plt.figure(figsize=(6.5, 6.6))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.set_extent(extent)
ax.add_feature(cartopy.feature.LAND, edgecolor='black')
ax.add_feature(cartopy.feature.RIVERS)
ax.add_feature(lakes_50m, facecolor='lightsteelblue',edgecolor='black')

# Plot altimetry points
ax.scatter(
    x=np.array(lake_winnipeg["longitude"]),
    y=np.array(lake_winnipeg["latitude"]),
    zorder=10,
    s=1
)

# Set plotting stuff.
plt.title(
    "Sentinel A ground tracks over Lake Winnipeg",
    fontsize=TITLE_SIZE,
    pad=20,
    color=TEXT_COLOUR
)
ax.text(-0.22, 0.55, 'Latitude (degrees)', va='bottom', ha='center',
        rotation='vertical', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=LABEL_SIZE, color=TEXT_COLOUR)
ax.text(0.5, -0.15, 'Longitude (degrees)', va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes, fontsize=LABEL_SIZE, color=TEXT_COLOUR)

# Format gridlines
# https://scitools.org.uk/cartopy/docs/latest/gallery/gridlines_and_labels/gridliner.html
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': TICK_LABEL_SIZE, 'color': TEXT_COLOUR}
gl.ylabel_style = {'size': TICK_LABEL_SIZE, 'color': TEXT_COLOUR}

plt.tight_layout()
fig.savefig(
    './out/sentinel_a_ground_track.png',
    dpi=400, 
    bbox_inches='tight', 
    transparent=True
)

####
#### Figure 2: Outlier Rejection
####
lake_water_mean = lake_winnipeg["lake_water_level"].mean()
lake_water_std = lake_winnipeg["lake_water_level"].std()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

# Plot histogram of the lake water levels
ax.hist(
    x=lake_winnipeg["lake_water_level"],
    bins=np.linspace(
        lake_water_mean - 5 * lake_water_std,
        lake_water_mean + 5 * lake_water_std,
        500
    )
)
ax.set_xlim(
    [
        lake_water_mean - 7 * lake_water_std,
        lake_water_mean + 7 * lake_water_std,
    ]
)

# Set shaded red areas, indicating what to exlude
ax.axvspan(
    xmin=ax.get_xlim()[0],
    xmax=lake_water_mean - 2 * lake_water_std,
    facecolor="#F02D3A",
    alpha=0.5
)
ax.axvspan(
    xmin=lake_water_mean + 2 * lake_water_std,
    xmax=ax.get_xlim()[1],
    facecolor="#F02D3A",
    alpha=0.5
)

# Add vertical lines at each standard deviation
vline_water_levels = [lake_water_mean + i * lake_water_std for i in range(-2,3)]
ax.set_ylim([0, 2000.])
ax.vlines(
    x=vline_water_levels,
    ymin=0., 
    ymax=2000., # hard code in 2000. here; setting ymax=ax.get_ylim()[1] isn't working properly
    color='k',
    linestyle='--',
    alpha=0.5
)

# Add thousands separator to y-axis
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
)

ax.set_xlabel(
    "Water level (m)",
    color=TEXT_COLOUR,
    fontsize=LABEL_SIZE,
    labelpad=20
)
ax.set_ylabel(
    "Number of data points",
    fontsize=LABEL_SIZE,
    labelpad=20,
    color=TEXT_COLOUR
)

# Add upper x-axis showing mu and +/- N standard deviations
ax1 = ax.twiny()
ax1.set_xlim(
    ax.get_xlim()
)
ax1.set_xticks(
    [lake_water_mean + i * lake_water_std for i in range(-2, 3)]
)
ax1.set_xticklabels(
    [
        r'-2$\sigma$',
        r'-1$\sigma$',
        r'$\mu$',
        r'1$\sigma$',
        r'2$\sigma$',
    ],
    fontsize=TICK_LABEL_SIZE,
    color=TEXT_COLOUR
)

# Set all x/y-ticks to have the right size and colour
for x in ax.get_xticklabels():
    x.set_fontsize(TICK_LABEL_SIZE)
    x.set_color(TEXT_COLOUR)
for y in ax.get_yticklabels():
    y.set_fontsize(TICK_LABEL_SIZE)
    y.set_color(TEXT_COLOUR)

# Add label saying to ignore the shaded red-ares    
ax.text(210, 1500, 'Ignore water\n' + r'levels of $\mu  \pm2\sigma$' + '\n(shaded)', color=TEXT_COLOUR, fontsize=13)

# Set the colour of the axes
ax1.spines["top"].set_color(TEXT_COLOUR)
ax1.spines["bottom"].set_color(TEXT_COLOUR)
ax1.spines["left"].set_color(TEXT_COLOUR)
ax1.spines["right"].set_color(TEXT_COLOUR)
ax.tick_params(axis='x', colors=TEXT_COLOUR)
ax.tick_params(axis='y', colors=TEXT_COLOUR)
ax1.tick_params(axis='x', colors=TEXT_COLOUR)

plt.title(
    "Distribution of lake water levels\nin Lake Winnipeg from Sentinel A",
    fontsize=TITLE_SIZE,
    pad=20,
    color=TEXT_COLOUR
)
plt.tight_layout()
fig.savefig(
    './out/outlier_rejection.png',
    dpi=400, 
    bbox_inches='tight', 
    transparent=True
)

####
#### Figure 3: Mean lake water levels on each day
####
# Read in the processed data; reuse the lake_winnipeg name for the data frame
lake_winnipeg = pd.read_csv("./processed/sentinel_a_lake_winnipeg_remove_outliers.csv")

# Reject outliers
lake_winnipeg = lake_winnipeg.loc[
    lake_winnipeg["reject"] == False
]

baseline_results = lake_winnipeg[
    [
        "date",
        "lake_water_level"
    ]
].groupby("date").agg(
    {
        "lake_water_level" : "median"
    }
).reset_index()
baseline_results.loc[:, "date_as_datetime"] = pd.to_datetime(baseline_results.loc[:, "date"], format="%Y%m%d")

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.plot(
    baseline_results["date_as_datetime"],
    baseline_results["lake_water_level"],
    linewidth=1.0,
    label="Median of each track",
    color="#F55536"
)
ax.set_xlabel(
    'Date',
    labelpad=20,
    color=TEXT_COLOUR,
    fontsize=LABEL_SIZE
)
ax.set_ylabel(
    'Lake Winnipeg water levels (m)',
    labelpad=20,
    color=TEXT_COLOUR,
    fontsize=LABEL_SIZE
)
plt.title(
    "Median of lake water levels on each day\nin Lake Winnipeg (Sentinel A)",
    fontsize=TITLE_SIZE,
    color=TEXT_COLOUR,
    pad=20
)
locator = matplotlib.dates.MonthLocator((1, 7))
fmt = matplotlib.dates.DateFormatter('%b-%Y')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(fmt)
for x in ax.get_xticklabels():
    x.set_rotation(45)
    x.set_fontsize(TICK_LABEL_SIZE)
    x.set_color(TEXT_COLOUR)

for y in ax.get_yticklabels():
    y.set_fontsize(TICK_LABEL_SIZE)
    y.set_color(TEXT_COLOUR)

# Set the colour of the axes
ax.spines["top"].set_color(TEXT_COLOUR)
ax.spines["bottom"].set_color(TEXT_COLOUR)
ax.spines["left"].set_color(TEXT_COLOUR)
ax.spines["right"].set_color(TEXT_COLOUR)
ax.tick_params(axis='x', colors=TEXT_COLOUR)
ax.tick_params(axis='y', colors=TEXT_COLOUR)
plt.tight_layout()
fig.savefig(
    './out/median.png',
    dpi=400, 
    bbox_inches='tight', 
    transparent=True
)

#####
##### Figure 4: Kalman filtered lake water levels; overlaid on median (figure 3) levels
#####
# Just read in the results; don't repeat processing here.
kalman_filtered = pd.read_csv("./processed/sentinel_a_lake_winnipeg_kalman_filtered.csv")

# Plot just the median
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.plot(
    baseline_results["date_as_datetime"],
    baseline_results["lake_water_level"],
    linewidth=1.0,
    label="Median of each track",
    alpha=0.5,
    color = "#F55536"
)
ax.plot(
    pd.to_datetime(kalman_filtered["date"], format="%Y%m%d"),
    kalman_filtered["kalman_filtered_lake_water_levels"],
    zorder=10,
    label="Kalman Filtered",
    alpha=0.85,
    color="#6BAB90"
)
ax.set_xlabel(
    'Date',
    labelpad=20,
    color=TEXT_COLOUR,
    fontsize=LABEL_SIZE,
)
ax.set_ylabel(
    'Lake Winnipeg water levels (m)',
    labelpad=20,
    color=TEXT_COLOUR,
    fontsize=LABEL_SIZE
)
plt.title(
    "Median and Kalman filtered lake water levels on each day\nin Lake Winnipeg (Sentinel A)",
    fontsize=TITLE_SIZE,
    color=TEXT_COLOUR,
    pad=20
)
locator = matplotlib.dates.MonthLocator((1, 7))
fmt = matplotlib.dates.DateFormatter('%b-%Y')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(fmt)
for x in ax.get_xticklabels():
    x.set_rotation(45)
    x.set_fontsize(TICK_LABEL_SIZE)
    x.set_color(TEXT_COLOUR)

for y in ax.get_yticklabels():
    y.set_fontsize(TICK_LABEL_SIZE)
    y.set_color(TEXT_COLOUR)

ax.legend(
)
# Set the colour of the axes
ax.spines["top"].set_color(TEXT_COLOUR)
ax.spines["bottom"].set_color(TEXT_COLOUR)
ax.spines["left"].set_color(TEXT_COLOUR)
ax.spines["right"].set_color(TEXT_COLOUR)
ax.tick_params(axis='x', colors=TEXT_COLOUR)
ax.tick_params(axis='y', colors=TEXT_COLOUR)
plt.tight_layout()
fig.savefig(
    './out/kalman.png',
    dpi=400, 
    bbox_inches='tight', 
    transparent=True
)

####
#### Figure 5: Neural network lake water levels
####
# Read in the data used to make the neural network figure
lake_winnipeg_nn = pd.read_csv("./processed/sentinel_a_b_lake_winnipeg_neural_network.csv")
fig = plt.figure(
    figsize=(6, 6)
)
ax = fig.add_subplot(111)
# Artificially split the time series into a train/test part
# Plot in-situ
ax.plot(
    pd.to_datetime(lake_winnipeg_nn["date"], format="%Y-%m-%d"),
    lake_winnipeg_nn["in_situ_lake_water_level"],
    color="#16E0BD",
    linestyle="-.",
    linewidth=1.0,
    label="In-situ lake water levels",
    alpha=0.9
)

# Plot median of lake water levels
ax.plot(
    pd.to_datetime(lake_winnipeg_nn["date"], format="%Y-%m-%d"),
    lake_winnipeg_nn["lake_water_level"],
    linestyle="-",
    linewidth=1.0,
    color="#F55536",
    alpha=0.5,
    label="Sentinel A/B lake water levels (median)"
)


ax.plot(
    pd.to_datetime(lake_winnipeg_nn["date"], format="%Y-%m-%d"),
    lake_winnipeg_nn["lake_water_levels_nn"], 
    color="#586A6A", # Deep space sparkle
    linestyle="--",
    linewidth=1.0,
    alpha=0.9,
    label="Neural Network"
)

locator = matplotlib.dates.MonthLocator((1, 7))
fmt = matplotlib.dates.DateFormatter('%b-%Y')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(fmt)
for x in ax.get_xticklabels():
    x.set_rotation(45)
    x.set_fontsize(TICK_LABEL_SIZE)
    x.set_color("w")
    
for y in ax.get_yticklabels():
    y.set_fontsize(TICK_LABEL_SIZE)
    y.set_color("w")
ax.legend()
ax.set_xlabel(
    "Date",
    labelpad=20,
    fontsize=LABEL_SIZE,
    color=TEXT_COLOUR
)
ax.set_ylabel(
    "Lake Water Level (m)",
    labelpad=20,
    fontsize=LABEL_SIZE,
    color=TEXT_COLOUR
)
ax.spines["top"].set_color(TEXT_COLOUR)
ax.spines["bottom"].set_color(TEXT_COLOUR)
ax.spines["left"].set_color(TEXT_COLOUR)
ax.spines["right"].set_color(TEXT_COLOUR)
ax.tick_params(axis='x', colors=TEXT_COLOUR)
ax.tick_params(axis='y', colors=TEXT_COLOUR)
plt.title(
    "Lake Winnipeg water levels: in-situ, median,\n and neural network",
    pad=20,
    fontsize=TITLE_SIZE,
    color=TEXT_COLOUR
)
plt.tight_layout()
fig.savefig(
    './out/neural_network.png',
    dpi=400, 
    bbox_inches='tight', 
    transparent=True
)