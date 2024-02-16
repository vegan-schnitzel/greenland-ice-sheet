"""
Define pdd model parameters.
"""
##########################
# TEMPERATURE FIELD [°C] #
# initial temperature value
T_INITIAL = 4

# randomness due to weather
# standard deviation of normal distribution
T_WEATHER_SCALE = 3.5

# elevation temperature decrease per 100m
LAPSE_RATE = 0.7

# latitudinal gradient
# absolute north-south difference
T_LATITUDE_DIFF = 15

# seasonal cycle
# absolute summer-winter difference
T_SEASONAL_DIFF = 20

##############################
# PRECIPITATION FIELD [mm/day]
# initial precipitation value
PR_INITIAL = 3.5

# latitudinal gradient
PR_LATITUDE_DIFF = 3.5

######################
# SURFACE MASS BALANCE
# pdd / melting factor [mm d-1 °C-1]
BETA = 5
