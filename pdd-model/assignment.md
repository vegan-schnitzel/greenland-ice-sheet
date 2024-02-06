This exercise has two parts:

1) Define (synthetic) fields for precipitation and temperature. You can think of the temperature field as a superposition of multiple influences:
   - a latitudinal gradient toward colder temperatures
   - a decrease of temperatures with elevation
   - a seasonal cycle
   - some randomness to represent weather.

The precipitation field is made in a similar way:
    - The water vapor content of the atmosphere, i. e., proximity to coast and temperature
    - Orographic precipitation creates some of the highest precipitation rates in the world

Remember that your final fields should have a daily resolution, i.e., a time axis of length 365 for the PDD method to work.

1) Calculate the surface mass balance based on these fields. This is again separated into a) accumulation, the precipitation that falls at temperatures below 0 degC, and b) ablation, using the PDD method. The total surface mass balance is then SMB = ACC + ABL.

### ToDo:
- find lapse rate value from literature
- implement sanity checks
- create an ocean mask by using elevation data
- seasonality of precipitation?
- write parameters of modul in section at the beginning of the code
- plot proper maps with coastlines
- use np.flip instead of origin=lower, so longitude is first dimension
- What about the units?

In the end we get a map of greenland with mass balances during one year.