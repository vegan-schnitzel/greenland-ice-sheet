"""
Define pdd model parameters.
"""
##########################
# TEMPERATURE FIELD [°C] #
# initial temperature value
T_INITIAL = 3.0

# randomness due to weather
# standard deviation of normal distribution
T_WEATHER_SCALE = 3.5

# elevation temperature decrease per 100m
# https://www.britannica.com/science/lapse-rate
LAPSE_RATE = 0.65

# latitudinal gradient
# absolute north-south difference
T_LATITUDE_DIFF = 15.0

# seasonal cycle
# absolute summer-winter difference
T_SEASONAL_DIFF = 2.0

##############################
# PRECIPITATION FIELD [mm/day]
# initial precipitation value
PR_INITIAL = 7.0

# latitudinal gradient
PR_LATITUDE_DIFF = 6.0

# weights of distance to coastline
PR_COASTLINE = 0.1

######################
# SURFACE MASS BALANCE
# pdd / melting factor [mm d-1 °C-1]
BETA = 5.0
