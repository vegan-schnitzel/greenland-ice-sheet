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
- [ ] add contours to input pdd fields and/or elevation contours?
- [ ] when outputing mean values, ocean isn't masked!
- [ ] implement sanity checks
- [ ] seasonality of precipitation?
- [ ] What about the units of pdd / melting factor?
- [ ] How to combine results of several simulations? Save data as npy?

In the end we get a map of greenland with mass balances during one year.

### Feedback:
Very nice with the table of effects!
The first three figures are not made by your code I think. Please cite clearly where they are from. 

Your SMB seems to be on a reasonable scale - that is good. It's a bit hard to see whether there is any melting at all, maybe try to plot the ocean in another color. It looks like it's very dominated by the latitudinal gradient, perhaps try to increase the effect of altitude as well. 

-- Also, is the latitudinal gradient the wrong way around? We would expect more precip in the south, but warmer temperatures. I think this is all solvable by a little tweaking of parameters here and there :)