import numpy as np
import matplotlib.pyplot as plt
import os

def warp_heading(heading):
    # if(heading >=0 and heading < 90):
    #     angle  = np.radians(90-heading)
    # elif(heading >= 90 and heading < 180):
    #     angle  = np.radians(heading-90)
    # elif(heading >= 180 and heading < 270):
    #     angle = np.radians(270-heading)
    # else:
    #     angle = np.radians(heading-270)
    return heading + 90

# Load data from file
data = np.loadtxt('mp3/waypoints/xy_demo.csv', delimiter=',')

# Extract x, y, and theta from data
x = data[:, 0]
y = data[:, 1]
theta = data[:, 2]
theta2 = np.copy(theta)
theta = [warp_heading(angle) for angle in theta2]
    

# Convert theta to radians
theta = np.deg2rad(theta)

# Compute arrow coordinates
dx = np.cos(theta)
dy = np.sin(theta)

# Create plot and set axis limits
fig, ax = plt.subplots()
ax.set_xlim([np.min(x) - 1, np.max(x) + 1])
ax.set_ylim([np.min(y) - 1, np.max(y) + 1])

# Plot arrows
ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)

# Show plot
plt.show()

