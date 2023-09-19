import numpy as np

# Constraints 
velocity_max = 0.8

# Controller COnstants
Kp = 0.6

def position_controller(cmd_state:np.ndarray,estimated_state:np.ndarray)->np.ndarray:
    state_error:np.ndarray = cmd_state - estimated_state
    state_error[2] = 0.0
    velocity = (state_error/np.linalg.norm(state_error)) * min(velocity_max, Kp*np.linalg.norm(state_error))
    return velocity 

