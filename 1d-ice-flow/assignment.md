
We want to implement a simple one-dimensional ice flow model based on the theory of lecture 5.
This tool will be very useful to understand the complex interactions of continental ice sheets with climate (lecture 3) and the solid earth below (future lecture).

You can choose a surface mass balance that is convenient for you. It does not have to be one from the previous exercises but the next exercise will extend this model to two dimensions so that you can use your "own" SMB.

#### The creep/flow parameter A. 

In our ice flux equations, we have the parameter A, which there has been some confusion about. This tells us how easily the ice deforms and depends on temperature and other properties of the ice. The larger A is, the quicker your ice flows. You can read about it's controls and find reasonable values for it in the Cuffey/Paterson book chapters 3.4.5 and 3.4.6. However, these values are sometimes a bit too small for our model, and you are free to deviate several orders of magnitude if you find that your ice flows very slowly (try A=e-22/18).

### Notes
- use step function for mass balance (-1 & 1)
  - later, use zonal mean or something else of Greenland
- set initial height to 0
- animate frames with several years difference

### ToDo
- [x] figure out cfl criteria
- [ ] create animation of results with matplotlib
- [ ] even if SMB is low and flow parameter in right change, almost no ice transport visible

