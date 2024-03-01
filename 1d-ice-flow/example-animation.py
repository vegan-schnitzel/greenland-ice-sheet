##### Interactive plotting example ####
import time
import matplotlib.pyplot as plt

plt.ion()

fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.set_ylim(-100,6000) # Nice to define limits, so the axes aren't changed all the time
height, = ax.plot(x,h) #x extent and h height of ice sheet

# while/for...
# updating ice dynamics
# The next part should be _inside_ this loop

 if year%30 == 0: # I update my plot every 30 years
      height.set_xdata(x)
      height.set_ydata(h)
      fig.canvas.draw()
      fig.canvas.flush_events()
      time.sleep(0.1) # Set pause between updating (remove if plot is slow)
      
  
plt.ioff() # This part comes after the whole loop is done