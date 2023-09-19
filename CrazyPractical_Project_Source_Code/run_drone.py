from drone import Drone
from cflib.utils import uri_helper
import cflib.crtp  # noqa
import numpy as np

uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E704')
cflib.crtp.init_drivers()

# Define the home position (X,Y) of take off pad
home = np.array([1.0,2.0])

# Define the cruise height
CRUISE_HEIGHT = 0.4

start_sweeping_position = np.array([3.5,3]) - home

# Run control
drone = Drone(uri,home_position=home, start_sweeping_position= start_sweeping_position, cruise_height=CRUISE_HEIGHT)