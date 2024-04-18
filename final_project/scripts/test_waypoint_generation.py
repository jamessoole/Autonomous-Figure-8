import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

# meters for discretizing line
DELTA_D = 0.1
# number of points to discretize
NUM_POINTS = 5

LEFT_BOX = 0
RIGHT_BOX = 1

UP = 0
DOWN = 1


def visualize_curve(x, y):
  tck,u = interpolate.splprep([x,y],s=100)
  unew = np.arange(0,1.01,0.01)
  out = interpolate.splev(unew,tck)
  plt.figure()
  plt.scatter(x,y)
  plt.plot(out[0],out[1])
  plt.show()


def get_circle_points(box_1_coordinates, box_2_coordinates, center_coordinates, car_heading, box=LEFT_BOX, num_points=10):
  unit_vec = np.array([np.cos(car_heading), np.sin(car_heading)])
  box_1_vec = box_1_coordinates - center_coordinates 
  
  cross_product = np.cross(unit_vec, box_1_vec)
  if cross_product > 0:
    left_box_coordinates, right_box_coordinates = box_1_coordinates, box_2_coordinates 
  else:
    left_box_coordinates, right_box_coordinates = box_2_coordinates, box_1_coordinates 

  # determine which box we are circling
  box_coordinates = left_box_coordinates if box==LEFT_BOX else right_box_coordinates

  radius = distance(box_coordinates,center_coordinates)

  angle_between_box_and_car = get_angle_between_points(box_coordinates, center_coordinates)

  angles = np.array([angle_between_box_and_car + 2 * np.pi / num_points * i for i in range(1,num_points+1)])
  headings = np.array([angle + np.pi / 2 for angle in angles])

  wps = np.array([np.array([box_coordinates[0] + np.cos(angle) * radius, box_coordinates[1] + np.sin(angle) * radius]) for angle in angles])

  xs = wps[:,0]
  ys = wps[:,1]

  if box == RIGHT_BOX:
    xs = np.roll(np.flip(xs), -1)
    ys = np.roll(np.flip(ys), -1)
    headings = np.roll(np.flip(headings) + np.pi, -1)
    

  print("X coordinates:")
  print(xs)
  print("Y coordinates:")
  print(ys)
  print("Headings:")
  print(headings)

  return xs, ys, headings



def get_angle_between_points(p1, p2):
  return np.arctan2((p1[1] - p2[1]),(p1[0] - p2[0]))


def distance(p1, p2):
  return ((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**0.5


# returns xs and ys as two arrays
def get_guiding_points(p1, p2, center, curr_x, curr_y, curr_yaw):
  # curr_x, curr_y, curr_yaw = self.get_gem_state()

  xs = np.array([curr_x + (np.cos(curr_yaw) * DELTA_D * i) for i in range(NUM_POINTS)])
  ys = np.array([curr_y + (np.sin(curr_yaw) * DELTA_D * i) for i in range(NUM_POINTS)])

  box_1_x = p1[0]
  box_1_y = p1[1]
  box_2_x = p2[0]
  box_2_y = p2[1]

  theta_1 = np.arctan2((box_1_y - box_2_y),(box_1_x - box_2_x)) + (np.pi / 2)
  theta_2 = theta_1 + np.pi

  step_1 = (center[0] + np.cos(theta_1), center[1] + np.sin(theta_1))
  step_2 = (center[0] + np.cos(theta_2), center[1] + np.sin(theta_2))

  distance_1 = (curr_x - step_1[0])**2 + (curr_y - step_1[1]**2)
  distance_2 = (curr_x - step_2[0])**2 + (curr_y - step_2[1]**2)

  theta = theta_1 if distance_1 < distance_2 else theta_2

  x_tails = np.array([center[0] + (np.cos(theta) * DELTA_D * i) for i in range(NUM_POINTS)])
  y_tails = np.array([center[1] + (np.sin(theta) * DELTA_D * i) for i in range(NUM_POINTS)])

  return np.concatenate((xs, x_tails)), np.concatenate((ys, y_tails))

def plot_arrows(x, y, theta, box_1, box_2, center, car_heading):
  dx = np.cos(theta)
  dy = np.sin(theta)

  box_x = np.array([box_1[0], box_2[0]])
  box_y = np.array([box_1[1], box_2[1]])

  all_x = np.concatenate((box_x, x))
  all_y = np.concatenate((box_y, y))
  
  # Create plot and set axis limits
  fig, ax = plt.subplots()
  ax.set_xlim([np.min(all_x) - 1, np.max(all_x) + 1])
  ax.set_ylim([np.min(all_y) - 1, np.max(all_y) + 1])

  # Plot arrows
  ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1)
  ax.quiver(center[0], center[1], np.cos(car_heading), np.sin(car_heading), color='green')
  ax.scatter(box_x, box_y, color='red')
  ax.scatter(center[0], center[1], color='orange')

  # Show plot
  plt.show()


def __main__():
  #xs, ys = get_guiding_points([0,0], [10,0], [5,0], -2, -10, np.pi)
  #visualize_curve(xs, ys)
  box_1 = np.array([-5,8])
  box_2 = np.array([0,8])
  center = np.array([-2.5,8])
  car_heading = np.pi/2
  box = RIGHT_BOX
  points = 10

  xs, ys, angles = get_circle_points(box_1, box_2, center, car_heading, box, points)
  plot_arrows(xs, ys, angles, box_1, box_2, center, car_heading)


__main__()



  
