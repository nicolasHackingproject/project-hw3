
import numpy as np
def pos_from_state(state):
  for i in range(0,10):
    column = (np.where(state[1,i,:] == 1.))
    if len(column[0]) > 0:
      column = column[0][0]
      row = i 
      return(column,row)

def state_from_pos(pos_x,pos_y,state):
  #Crop to keep a window [49,4]
  if pos_y > 8:
    y_max = 9
    y_min = 5
  elif pos_y < 2:
    y_max = 4
    y_min = 0
  else:
    y_max = pos_y + 2
    y_min = pos_y - 2
  
  if pos_x > 45:
    x_max = 49
    x_min = 40
  elif pos_x < 5:
    x_max = 9
    x_min = 0
  else:
    x_max = pos_x + 4
    x_min = pos_x - 5

  return(state[:,y_min:(y_max+1),x_min:(x_max+1)])


 # Crop initial state
pos_y,pos_x = pos_from_state(state)
state = state_from_pos(pos_x,pos_y,state)
