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
#    - Read in Sentinel A and B data.
#      Sentinel B data is used only for our neural network.
#    - Figure 1: Sentinel A ground tracks over Lake Winnipeg
#    - Figure 2: Outlier rejection (reject points +/- 2*sigma from the mean)

# Constants for plotting
FIG_WIDTH_INCHES = 6.5
FIG_HEIGHT_INCHES = 6.5
TITLE_SIZE = 15
TICK_LABEL_SIZE = 14
LABEL_SIZE = 14
TEXT_COLOUR="w"

#### Read in Sentinel A and B data and isolate Lake Winnipeg data.
def read_sentinel_data(filename):
    sentinel_data = pd.read_csv(filename)
    sentinel_data = sentinel_data.rename(
        columns={
            "Date (YYYYMMDD)" : "date",
            "Lake_name" : "lake_name",
            "Latitude" : "latitude",
            "Longitude" : "longitude",
            "Relaive_orbit" : "relative_orbit",
            "Lake water level (m)" : "lake_water_level"
        }
    )
    # Convert date to date time.
    sentinel_data.loc[:, "date"] = pd.to_datetime(sentinel_data.loc[:, "date"], format="%Y%m%d")
    return sentinel_data

# Sentinel A data is used for our initial data exploration including Kalman filtering
# Sentinel B data is used along with Sentinel A data for our neural network.
sentinel_data_A = read_sentinel_data("data/Sentinel_3A_water_level_Version0.csv")
sentinel_data_B = read_sentinel_data("data/Sentinel_3B_water_level_Version0.csv")

# Isolate Lake Winnipeg data from Sentinel A data
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
    transparent=False
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





















# Concatenate Sentinel A and B data, and isolate Lake Winnipeg data
sentinel_data = pd.concat([sentinel_data_A, sentinel_data_B])
lake_winnipeg = sentinel_data[
    sentinel_data["lake_name"] == "Winnipeg"
]

#### Read in-situ data and join it to the Sentinel data
lake_winnipeg_in_situ = pd.read_csv("./data/WinnipegLake_at_GeorgeIsland.csv")
lake_winnipeg_in_situ = lake_winnipeg_in_situ.rename(
    columns={
        "Date" : "date",
        "Value (m)" : "in_situ_lake_water_level"
    }
)
# Convert date to date time and select only date and in_situ_lake_water_level columns
lake_winnipeg_in_situ.loc[:, "date"] = pd.to_datetime(lake_winnipeg_in_situ.loc[:, "date"], format="%Y-%m-%d")
lake_winnipeg_in_situ = lake_winnipeg_in_situ[["date", "in_situ_lake_water_level"]]

# Join the data on date
lake_winnipeg = lake_winnipeg.merge(lake_winnipeg_in_situ, on='date', how='left')


