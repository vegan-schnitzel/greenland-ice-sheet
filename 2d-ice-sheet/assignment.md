The final assignment will summarize most of what we learned in this class. It builds on all previous assignments and you will be able to use the code that you already have. This work counts for 80% of the final grade.

The objective is to write a 2D model for the Greenland ice sheet and to simulate how different SMBs impact the simulations. It is a relatively small step from the 1D model. Your report should contain three parts:

1) Document the performance of the 2D ice sheet model on flat bedrock. This is a benchmark simulation to ensure we can trust the results and catch possible mistakes and so we should all use the same set up:

   - The domain should have a resolution of 40 km, see instructions video. Input file: `grl40_bed_relaxed.npy`
   - The SMB is -2 m/yr everywhere outside a 10-by-10 grid box square in the middle of the domain. There the SMB should be 2 m/yr.
   - The report should include 2D maps of ice thickness but you can also include sections through the simulated ice sheet.

2) Simulate the Greenland ice sheet for idealized climate anomalies, i.e., combinations of warmer/colder and wetter/dryer than normal climates. Use these simulations to answer the following questions:

    How do different climate anomalies impact the size/shape/stability of the Greenland ice sheet?
    How do different regions on Greenland respond differently and why?
    Can you decompose the effects of anomalous temperature and precipitation? Section 5 of this paper 

    Download this paper gives an example of what is possible.

You can choose yourself whether to run these simulations using the present-day ice topography or ice-free conditions, or both. Just document and justify your choice clearly. Consider the following:

    You will need significantly more than three simulations because things do not usually work out the first (or second) time. In addition, each simulation takes up to half an hour to complete. Do not start too late.
    If you start your simulation from an ice-free state, i.e., only bedrock, try using a SMB that is only a little positive. As the ice grows, the positive elevation feedback will make the SMB increasingly more positive.

 

3) How does the Greenland ice sheet respond to climate warming until the year 3000?

You should start your simulation with the modern ice thickness and

    run your own global warming scenario (partial overlap with point 2 above).
    use the provided forcing fields that are based on a simulation with the Norwegian Earth System Model (NorESM). You can find the files in the "Files" section and an example script of how to use them in the python script climate_forcing_example.py 

    Download climate_forcing_example.py.
    submit all simulation results in time so that the PhD student in our group can summarize the findings (June 5).

 

4) Bonus (not mandatory): What are you curious about? Some ideas:

    Does it work at a resolution higher than 40 km? I also have climate data on the 10 km grid, but it was too large for mittUiB. Ask if you're interested.
    The effect of bedrock sinking (lecture 10)
    The effect of basal hydrology (lecture 8)
    Can we see ice sheet bistability in this model (hysteresis, lecture 10)?
    How and how fast does the ice sheet respond to transient climate change (lecture 3)?
    Your ideas. Get in touch if you want to discuss or if you are unsure how to implement a certain process in code.

Grading scheme: https://mitt.uib.no/courses/45721/pages/term-paper-grading-scheme