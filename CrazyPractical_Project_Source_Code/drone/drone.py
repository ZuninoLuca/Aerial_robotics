import time
import numpy as np
import pandas as pd
from typing import Dict, List
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from sklearn.decomposition import PCA
from collections import deque

from drone.grid_map import GridMap, calc_obstacle_map
from .discretizing import get_discretized_path, sweep_coverage_points, home_point_coverage

import matplotlib.pyplot as plt

deque_max = 30

class Drone:
    def __init__(self, link_id:str, home_position:np.ndarray,start_sweeping_position: np.ndarray, cruise_height = 0.5) -> None:
        # Define the drone
        self._cf = Crazyflie(rw_cache=".cache")
        self._link_id = link_id

        # logging file
        self.file_object = open('logging.txt', 'a')

        # Define the home position to go to
        self.home_position = home_position

        # special waypoint 
        self.start_sweeping_position = start_sweeping_position

        # Set a certain cruise height
        self.cruise_height = cruise_height

        # Add connection callback
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)
    
        # Store the current position and sensor values
        self.current_position:np.ndarray = np.zeros((3,1), dtype="float")
        self.sensor_measures:Dict = {}
        self.sensor_measures_list = deque([], maxlen=5)

        # peak detection variables
        self.start_landing_movement = False
        self.start_peak_detecting = False
        self.adjust_landing_movement = False
        self.landing_movement_adjusted = False
        self.landing_threshold = 0.06
        self.landing_attempts = 0

        # forward path variables
        self.forward_path:List[np.ndarray] = []
        self.save_forward_path = False
        self.forward_path_time = 0
        self.entered_obstacle_avoid = False

        # calculation of start position
        self.calculate_start_point = False
        self.best_start_point:Dict[str,List] = {"start_point":[]}
        
        # direction of motion calculation
        self.start_direction_of_motion_calculation = False

        # current position 
        self.current_pos_list = deque([], maxlen=deque_max)

        # obstacle detect threshold
        self.threshold = 600

        # Initialize Grid Map
        self.grid_map:GridMap = GridMap(width = 50, height=30, center_x= 2.5,center_y=1.5, resolution=0.1)
        self.landed_on = None

        # Calculate landing pad center
        self.detected_pad_borders:Dict[int, List[np.ndarray]] = {0:[],1:[],2:[]}
        self.landing_points_count = 4

        # Execute the mission
        self.run()

        # close link after completion
        self._cf.close_link()
        self.file_object.close()
 

    def compute_trajectory(self):
        print(self.sensor_measures["left"],(1000*(3-self.current_position[1]))/2, (self.sensor_measures["right"]),(1000*self.current_position[1])/2)
        if self.sensor_measures["left"]>(1000*(3-self.current_position[1]))/2 and (self.sensor_measures["right"]>(1000*self.current_position[1]))/2:
            print("APPLY SWEEP")

            # sweep points
            sweep_points = [np.array([0,0]),np.array([0, 3-self.home_position[1]]),np.array([0,-self.home_position[1]])]

            # Initial found optimal
            found_optimal = False

            # start data acquisition
            self.calculate_start_point = True

            while len(sweep_points)>1 and not found_optimal:
                sweep_point = sweep_points.pop(0)
                path_to_follow = get_discretized_path(sweep_point, sweep_points[0], step_size=0.1)
                for point in path_to_follow:
                    self.position_commander.go_to(point[0], point[1], self.cruise_height, velocity=0.8)
                    start_point,found_optimal = get_optimal_forward_path(self.best_start_point["start_point"])
                    if found_optimal:
                        print("FOUND OPTIMAL: ", found_optimal)
                        break
            
            # stop data acquisition
            self.calculate_start_point = False
            
            if not found_optimal:
                start_point, _ = get_optimal_forward_path(self.best_start_point["start_point"])

            # adjust start point factor 
            way_points = [start_point, start_point + np.array([1.8,0.0]), start_point + np.array([3.5-self.home_position[0],0.0]), self.start_sweeping_position]
            
            self.way_points = sweep_coverage_points(self.cruise_height, way_points, self.home_position[0], self.home_position[1])
            self.original_waypoints = self.way_points
        else:
            print("SWEEP NOT POSSIBLE")
            way_points = [np.array([0,0]),np.array([1.8,0]),np.array([3.5-self.home_position[0],0]), self.start_sweeping_position]
            self.way_points = sweep_coverage_points(self.cruise_height, way_points, self.home_position[0], self.home_position[1])
            self.original_waypoints = self.way_points

    def run(self):
        # finish first sequence
        land_detected = False
        obstacle_avoid = False

        with SyncCrazyflie(self._link_id,self._cf) as scf:
            # reset kalman filter values
            self._cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            self._cf.param.set_value('kalman.resetEstimation', '0')
            time.sleep(2)

            # define position commander
            self.initialize_position_commander(scf)

            # take off
            self.take_off()
            self.compute_trajectory()
            self.start_estimate_direction_of_motion()

            # start saving the forward path
            self.save_forward_path = True
            self.forward_path_time = time.time()

            # Forward pass
            while (self.landing_attempts < 2) and not land_detected:
                while len(self.way_points)>1 and not land_detected:
                    
                    path_to_follow = self.get_local_path(obstacle_avoid)
                    obstacle_avoid = self.follow_path(path_to_follow)
                    
                    if self.adjust_landing_movement:
                        land_detected = self.localize_landing_pad_and_land()
                
                if not land_detected: 
                    self.landing_attempts += 1
                    print("LANDING PAD not DETECTED")
                    self.attempt_landing_again()

            
            if land_detected:   
                self.stop_estimate_direction_of_motion()
                self.view_mission_results()
                
                # sleep time is 3 seconds 
                time.sleep(3)

                # time forward function
                print("Time elapsed: ", self.forward_path_time)
                self.take_off()

                # take off and compute way back
                self.compute_way_back(from_landing_pad=True)

            # Landing pad not found in two attempts
            else:
                # Follow a_star path back
                print(" GIVING UP !")
                self.compute_way_back(from_landing_pad=False)

            self.start_estimate_direction_of_motion()

            # Backward pass
            while len(self.way_points)>0:
                way_point = self.way_points.pop(0)
                self.position_commander.go_to(x =way_point[0], y=way_point[1], z=self.cruise_height)
                triggered_dir, detected, safety_detect = self.check_obstacle()
                if detected or safety_detect:
                    if detected:
                        self.obstacle_avoid(triggered_dir)
                    self.compute_way_back(from_landing_pad=False) 
            
            # land on pad
            self.localize_home_position()
            self.view_mission_results()

    def attempt_landing_again(self):
        print("LANDING PAD not DETECTED")
        # Reset the waypoints to start sweeping again
        way_points = [self.start_sweeping_position]
        self.way_points = sweep_coverage_points(self.cruise_height, way_points, self.home_position[0], self.home_position[1], expand_border=True)

        # Reset all the flags
        self.detected_pad_borders = {0:[],1:[],2:[]}
        self.start_peak_detecting = True
        self.adjust_landing_movement = False
        self.landing_movement_adjusted = False
        self.start_landing_movement = False
        return 
    
    def give_up(self):
        self.detected_pad_borders = {0:[],1:[],2:[]}
        self.start_peak_detecting = False
        self.adjust_landing_movement = False
        self.landing_movement_adjusted = False
        self.start_landing_movement = False
        return 

    def follow_forward_path(self):
        # Reverse the points 
        self.way_points = self.forward_path
        self.way_points.reverse()

    def localize_home_position(self):
        # border detection
        self.detected_pad_borders = {0:[],1:[],2:[]}
        self.start_peak_detecting = False
        self.adjust_landing_movement = False
        self.landing_movement_adjusted = False
        self.start_landing_movement = False

        # get landing pad waypoints
        land_detected = False
    
        # sweep region
        self.way_points = home_point_coverage(width=1.0, starting=0.5, height=1.5)
        
        self.start_peak_detecting = True
        # index of popping to have made a L-shape
        index = 0
        while len(self.way_points)>2 and not land_detected: 
            self.position_commander.go_to(self.way_points[index%2][0], self.way_points[index%2][1], self.cruise_height, velocity=0.4)
            index+=1
            if index==2:
                index = 0
                self.way_points.pop(0)
                self.way_points.pop(0)

            if self.adjust_landing_movement:
                land_detected = self.localize_landing_pad_and_land()
                

        if not land_detected:
            print("LANDING not DETECTED")
            home_x, home_y = best_know_home_position(self.detected_pad_borders)
            self.position_commander.go_to(x = home_x, y = home_y, z=self.cruise_height)
            self.land_on_pad()
        
    def start_estimate_direction_of_motion(self):
        self.start_direction_of_motion_calculation = True

    def stop_estimate_direction_of_motion(self):
        self.start_direction_of_motion_calculation = False

    def initialize_position_commander(self, crazyflie):
        self.position_commander = PositionHlCommander(crazyflie, default_height=self.cruise_height)

    def compute_way_back(self, from_landing_pad = False):
        # take off position 
        if from_landing_pad:
            start_x = self.landed_on[0] + self.home_position[0]
            start_y = self.landed_on[1] + self.home_position[1]

        else:
            start_x = self.current_position[0] + self.home_position[0]
            start_y = self.current_position[1] + self.home_position[1]
        
        print("start x: ", start_x)
        print("start y: ", start_y)

        try:
            new_x_waypoints, new_y_waypoints = self.grid_map.planning(self.home_position[0], self.home_position[1], start_x, start_y)
            self.way_points = get_return_waypoints(new_x_waypoints, new_y_waypoints, self.home_position)
            # append 
            self.way_points.append(np.array([0.0,0.0]))
            print(self.way_points)
        except:
            print("OPTIMAL way back calculation failed")
            if not from_landing_pad:
                start_x = self.current_position[0]
                start_y = self.current_position[1]
            else:
                start_x = self.landed_on[0] 
                start_y = self.landed_on[1] 
                
            self.way_points = get_discretized_path(np.array([start_x, start_y]), np.array([0,0]), 0.1)
            print(self.way_points)
            pass

    def estimate_feasible_sweep_region(self):
        # read front and back sensor and estimate the height
        front_sensor, back_sensor = self.sensor_measures["front"], self.sensor_measures["back"]
        height = min(front_sensor/1000 + back_sensor/1000,1)

        # calculate the minimum box
        min_right_sensor, min_left_sensor = self.sensor_measures["right"], self.sensor_measures["left"]
        start_point_x = min(back_sensor/2000 + self.current_position[0], 1.5-self.home_position[0])
        sweep_points = [np.array([start_point_x,0]), np.array([start_point_x-height,0])]

        for waypoint in sweep_points:
            self.position_commander.go_to(waypoint[0], waypoint[1], self.cruise_height)
            min_right_sensor = self.sensor_measures["right"] if self.sensor_measures["right"]<min_right_sensor else min_right_sensor
            min_left_sensor = self.sensor_measures["left"] if self.sensor_measures["left"]<min_left_sensor else min_left_sensor

        width = min((min_left_sensor + min_right_sensor)/2000, 1.2)

        print(width)
        return start_point_x, height, width

    def view_mission_results(self):
        self.grid_map.plot_grid_map()
        plt.savefig("mission_result.png")

    def execute_landing_sequence(self):
        self.start_peak_detecting = False
        landing_movement = get_required_points(self.detected_pad_borders, False)
        print("LAND OVER", landing_movement)
        self.landed_on = landing_movement
        self.position_commander.go_to(x = landing_movement[0], y=landing_movement[1], velocity=0.2)
        self.land_on_pad()
        return

    def localize_pad_sequence(self):
        adjusted_movement = get_required_points(self.detected_pad_borders, True)
        print("HOVER OVER", adjusted_movement)        
        self.update_motion_for_landing(adjusted_movement)
        return

    def update_motion_for_landing(self, adjusted_movement):
        self.original_waypoints = self.way_points

        self.way_points = [np.array([adjusted_movement[0]+0.3,adjusted_movement[1]]), np.array([adjusted_movement[0]-0.3,adjusted_movement[1]]), np.array([adjusted_movement[0]+0.3,adjusted_movement[1]])]
        
        # go to hovering position
        way_point = self.way_points.pop(0)
        self.position_commander.go_to(way_point[0], way_point[1], z=self.cruise_height, velocity=0.2)
        self.landing_movement_adjusted = True

        while len(self.way_points)>1:                
            way_point = self.way_points.pop(0)
            self.position_commander.go_to(way_point[0], way_point[1], z=self.cruise_height, velocity=0.2)    

        self.way_points = self.original_waypoints

    def localize_landing_pad_and_land(self):
        self.localize_pad_sequence()
        print("START LANDING MOVEMENT", self.start_landing_movement)
        if self.start_landing_movement:
            self.execute_landing_sequence()
            return True
        else:
            self.adjust_landing_movement = False
            self.landing_movement_adjusted = False
            self.detected_pad_borders = {0:[],1:[],2:[]}
            return False

    def follow_path(self,path_to_follow: List[np.ndarray]):
        for point in path_to_follow:
            # Follow a discritized step
            self.position_commander.go_to(point[0],point[1], self.cruise_height)

            # check if start landing sequence:
            if self.adjust_landing_movement: 
                return False

            # Check if obstacles exist and avoid them
            triggered_dir, detected, safety_detected = self.check_obstacle()
            if detected or safety_detected:
                if detected:
                    self.obstacle_avoid(triggered_dir)
                return True

        self.way_points.pop(0)
        return False


    def get_local_path(self, obstacle_avoid):
        # Get current waypoint and next and calculate the discretized path
        if obstacle_avoid:
            current_waypoint = self.current_position[:2]
            next_waypoint = self.get_next_obstacle_avoid_waypoint()
        else:
            current_waypoint = self.way_points[0]
            next_waypoint = self.way_points[1]

        if (current_waypoint[0] == self.start_sweeping_position[0]):
            self.start_peak_detecting = True

        if self.start_peak_detecting:
            path_to_follow = get_discretized_path(current_waypoint,next_waypoint, step_size=0.05)
        else:
            path_to_follow = get_discretized_path(current_waypoint,next_waypoint, step_size=0.1)
        return path_to_follow

    def get_next_obstacle_avoid_waypoint(self):
        while self._validate_checkpoint(cmd_state=self.way_points[1],radius=0.9):
            print("WAY POINT VALIDATED")
            self.way_points.pop(0)
        return self.way_points[1]

    def _validate_checkpoint(self, cmd_state:np.ndarray, radius:float):
        error = cmd_state - self.current_position[:2]
        checkpoint_validated = np.linalg.norm(error)<radius

        print("validate error:",np.linalg.norm(error))

        if (cmd_state[0] == self.start_sweeping_position[0]) and checkpoint_validated:
            self.start_peak_detecting = True
        return checkpoint_validated

    def land_on_pad(self):
        self.position_commander.down(0.3)
        time.sleep(1.0)
        self.position_commander.land()

    def take_off(self):
        self.position_commander.take_off(self.cruise_height)   
        time.sleep(1.0)

    def peak_detect(self):      # Find landing area 
        if not self.landing_movement_adjusted: 
            direction_of_motion = 1 if (self.way_points[1][1] - self.way_points[0][1])>0 else 0
        else:
            direction_of_motion = 2
        if (self.current_position[2] - self.cruise_height)>self.landing_threshold: 
            if len(self.detected_pad_borders[direction_of_motion])<=self.landing_points_count:
                self.detected_pad_borders[direction_of_motion].append(self.current_position[:2])
            print("0: ",len(self.detected_pad_borders[0]), "1: ",len(self.detected_pad_borders[1]),"2: ",len(self.detected_pad_borders[2]) )
            self.check_start_landing_movement()

    def check_start_landing_movement(self):
        if len(self.detected_pad_borders[0]) >= self.landing_points_count and len(self.detected_pad_borders[1]) >= self.landing_points_count:
            self.adjust_landing_movement = True
            if  len(self.detected_pad_borders[2]) >= self.landing_points_count: 
                self.start_landing_movement = True

    def check_obstacle(self):
        detect = False
        triggered_dir = None
        safety_detect = False

        if(len(self.current_pos_list) == deque_max):
            direction_vector = self.current_pos_list[deque_max-1]-self.current_pos_list[0]
            position_data = np.array(self.current_pos_list)
            pca = PCA(n_components=1).fit(position_data)

            motion_vector = np.array(pca.components_)
            dir = np.dot(np.squeeze(direction_vector),np.squeeze(motion_vector))
            if(dir<0):
                motion_vector = -motion_vector

            u_f = np.array([1,0])
            u_r = np.array([0,-1])
            u_b = np.array([-1,0])
            u_l = np.array([0,1])

            dot_f = np.dot(u_f, motion_vector.reshape(2))
            dot_r = np.dot(u_r, motion_vector.reshape(2))
            dot_b = np.dot(u_b, motion_vector.reshape(2))
            dot_l = np.dot(u_l, motion_vector.reshape(2))

            threshold_f = self.threshold*dot_f if dot_f>0 else 0
            threshold_r = self.threshold*dot_r if dot_r>0 else 0 
            threshold_b = self.threshold*dot_b if dot_b>0 else 0 
            threshold_l = self.threshold*dot_l if dot_l>0 else 0

            #print(f"front: {threshold_f}:{self.sensor_measures['front']} , left: {threshold_l}:{self.sensor_measures['left']}, back: {threshold_b}:{self.sensor_measures['back']}, right: {threshold_r}:{self.sensor_measures['right']}")
            triggered_dir, detect = self.get_highest_sensor_value([threshold_f,threshold_l,threshold_b,threshold_r])
            safety_direction, safety_detect = self.provide_safety_margin([threshold_f,threshold_b, threshold_l, threshold_r])
            if safety_direction == triggered_dir:
                detect = False
        return triggered_dir, detect, safety_detect

    def provide_safety_margin(self, thresholds, cutoff_threshold = 155):
        direction_map = ["front","back","left","right"]
        detected_direction = None
        safety_detect = False
        for index, threshold in enumerate(thresholds):
            if threshold<cutoff_threshold and self.sensor_measures[direction_map[index]]<threshold:
                print("PROVIDED SAFETY MARGIN")
                safety_detect = True
                detected_direction = direction_map[index]
                if index == 0 or index == 2:
                    self.obstacle_avoid_motion(direction_map[index+1], 0.2)
                else:
                    self.obstacle_avoid_motion(direction_map[index-1], 0.2)

        return detected_direction, safety_detect

    def get_available_direction_of_travel(self,A,B):
        A_meas = self.sensor_measures[A]
        B_meas = self.sensor_measures[B]
        return A_meas>self.threshold, B_meas>self.threshold

    def obstacle_avoid(self,triggered_direction):
        print("INSIDE OBSTACLE AVOIDANCE", self.way_points)
        forward_motion = 0.15
        
        #choose neighbor 1 and neigbor 2 
        neighbor_1,neighbor_2 = get_neighbor_sensors(triggered_direction)

        # Initially both 
        neighbor_1_available = True
        neighbor_2_available = True
        
        # Direction followed
        dir_to_check = None
        position_error_vector = None

        while (self.sensor_measures[triggered_direction] < self.threshold):
            self.entered_obstacle_avoid = True

            neighbor_1_available, neighbor_2_available = self.get_available_direction_of_travel(neighbor_1,neighbor_2)
            print("++++++++++++AVAILABLE SENSORS+++++++++")
            print(neighbor_1, neighbor_1_available)
            print(neighbor_2, neighbor_2_available)
            print("+++++++++++++++++++++++++++++++++++++")
            print("STEP 1:RUNNING")
            # If left is availble and not right
            if (neighbor_1_available and not neighbor_2_available):
                self.obstacle_avoid_motion(neighbor_1,0.02)
                dir_to_check = neighbor_2
            
            # If right is available and not left
            elif (not neighbor_1_available and neighbor_2_available):
                self.obstacle_avoid_motion(neighbor_2,0.02)
                dir_to_check = neighbor_1

            # If both are available, favor the one that will keep us in the middle of the arena
            elif (neighbor_1_available and neighbor_2_available):
                if(position_error_vector is None):
                    position_error_vector =  (self.way_points[1] - self.current_position[:2])

                    if(triggered_direction == "front" or triggered_direction == "back"):
                        if ((self.current_position[1] + self.home_position[1]) < 0.1):
                            position_error_vector = np.array([0,0.1])
                            print("CORNER POINT-RIGHT")
                        elif((self.current_position[1] + self.home_position[1]) > 2.9):
                            position_error_vector = np.array([0,-0.1])
                            print("CORNER POINT-LEFT")

                dir_to_check = self.follow_optimal_obstacle_avoid_path_when_both_available(triggered_direction, position_error_vector)
                
            else:
                print("NEED TO HANDLE THIS CASE")
                time.sleep(2)
                #dir_to_check = self.follow_optimal_obstacle_avoid_path_none_is_available(triggered_direction)

        if dir_to_check is not None:
            print("STEP 2:RUNNING")
            self.obstacle_avoid_motion(triggered_direction, 0.6)  
            while (self.sensor_measures[dir_to_check]<self.threshold):
                print("STEP 2:RUNNING")
                print("dir to check:", dir_to_check, self.sensor_measures[dir_to_check])
                self.obstacle_avoid_motion(triggered_direction, forward_motion)
            return 

    def follow_optimal_obstacle_avoid_path_when_both_available(self, triggered_direction, position_error):

        if triggered_direction == "front":
            self.position_commander.left(0.02) if position_error[1]>0 else self.position_commander.right(0.02)
            dir_to_check = "right"  if position_error[1]>0 else "left"

        elif triggered_direction =="left":
            self.position_commander.forward(0.02) if position_error[0]>0 else self.position_commander.back(0.02)
            dir_to_check = "back"  if position_error[0]>0 else "front"
            print("left error", position_error)

        elif triggered_direction == "right":
            self.position_commander.forward(0.02) if position_error[0]>0 else self.position_commander.back(0.02)
            dir_to_check = "back"  if position_error[0]>0 else "front"
            print("right error", position_error)

        elif triggered_direction == "back":
            self.position_commander.left(0.02) if position_error[1]>0 else self.position_commander.right(0.02)
            dir_to_check = "right"  if position_error[1]>0 else "left"

        return dir_to_check

    def follow_optimal_obstacle_avoid_path_none_is_available(self, triggered_direction):
        if triggered_direction == "front":
            self.position_commander.back(0.02)
            dir = "front"
        elif triggered_direction =="left":
            self.position_commander.right(0.02)
            dir = "left"
        elif triggered_direction == "right":
            self.position_commander.left(0.02)
            dir = "right"
        elif triggered_direction == "back":
            self.position_commander.forward(0.02)
            dir = "back"
        return dir   
    
    def get_highest_sensor_value(self, thresholds):
        max_margin_measure = 0
        direction_map = ["front", "left","back", "right"]
        detect = False
        triggered_dir = None
        for direction, threshold in zip(direction_map, thresholds):
            if(self.sensor_measures[direction] < threshold):
                detect = True
                # find detected regions
                if((threshold - self.sensor_measures[direction]) > max_margin_measure):
                    max_margin_measure = threshold-self.sensor_measures[direction] 
                    triggered_dir = direction
                    
        return triggered_dir, detect
                
    def obstacle_avoid_motion(self,follow, distance):
        obstacle_avoid_move={"front":self.position_commander.forward, "left": self.position_commander.left,"back": self.position_commander.back, "right": self.position_commander.right}
        obstacle_avoid_move[follow](distance)
        return 

    def pos_data(self, timestamp, data, logconf):
        position:np.ndarray = np.array([
            data['stateEstimate.x'],
            data['stateEstimate.y'],
            data['stateEstimate.z']
        ])
        self.current_position = position
        if self.start_direction_of_motion_calculation:
            self.current_pos_list.append(self.current_position[:2])

        self.grid_map.set_value_from_xy_pos(self.current_position[0]+self.home_position[0], self.current_position[1]+self.home_position[1], 1)
        if self.start_peak_detecting:
            self.forward_path_time = time.time() - self.forward_path_time
            self.save_forward_path = False
            self.peak_detect()

        if self.save_forward_path:
            self.forward_path.append(self.current_position[:2])
        return 

    def check_obstacle_and_add(self,sensor_value,sensor_name):
        if sensor_value<600:
            extra = sensor_value/1000
            width = 0.4
            current_position = self.current_position[0:2] + self.home_position

            if(sensor_name == "front"):
                calc_obstacle_map(self.grid_map,current_position[0]+extra,current_position[1],width)
            elif(sensor_name == "back"):
                calc_obstacle_map(self.grid_map,current_position[0]-extra,current_position[1],width)
            elif(sensor_name == "left"):
                calc_obstacle_map(self.grid_map,current_position[0],current_position[1]+extra,width) 
            elif(sensor_name == "right"):
                calc_obstacle_map(self.grid_map,current_position[0], current_position[1]-extra,width) 

    def meas_data(self, timestamp, data, logconf):
        measurement = {
            'roll': data['stabilizer.roll'],
            'pitch': data['stabilizer.pitch'],
            'yaw': data['stabilizer.yaw'],
            'front': data['range.front'],
            'back': data['range.back'],
            'up': data['range.up'],
            'down': data['range.zrange'],
            'left': data['range.left'],
            'right': data['range.right']
        }
        self.sensor_measures_list.append(measurement)
        self.sensor_measures =  get_average_sensor_measurements(self.sensor_measures_list)

        sensors_of_interest = ["front", "back","left","right"]
        if self.start_direction_of_motion_calculation and not self.adjust_landing_movement:
            for sensor in sensors_of_interest:
                self.check_obstacle_and_add(self.sensor_measures[sensor],sensor)

        if self.calculate_start_point:
            self.best_start_point["start_point"].append((measurement["front"], self.current_position[:2]))

        self.file_object.write(f"front: {self.sensor_measures['front']} left: {self.sensor_measures['left'],} right:{self.sensor_measures['right']} \n")
        
        return 

    def _connected(self, URI):
        print('We are now connected to {}'.format(URI))

        # The definition of the logconfig can be made before connecting
        lpos = LogConfig(name='Position', period_in_ms=10)
        lpos.add_variable('stateEstimate.x')
        lpos.add_variable('stateEstimate.y')
        lpos.add_variable('stateEstimate.z')

        try:
            self._cf.log.add_config(lpos)
            lpos.data_received_cb.add_callback(self.pos_data)
            lpos.error_cb.add_callback(self._log_error)
            lpos.start()

        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Position log config, bad configuration.')

        lmeas = LogConfig(name='Meas', period_in_ms=10)
        lmeas.add_variable('range.front')
        lmeas.add_variable('range.back')
        lmeas.add_variable('range.up')
        lmeas.add_variable('range.left')
        lmeas.add_variable('range.right')
        lmeas.add_variable('range.zrange')
        lmeas.add_variable('stabilizer.roll')
        lmeas.add_variable('stabilizer.pitch')
        lmeas.add_variable('stabilizer.yaw')

        try:
            self._cf.log.add_config(lmeas)
            lmeas.data_received_cb.add_callback(self.meas_data)
            lmeas.error_cb.add_callback(self._log_error)
            lmeas.start()

        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Measurement log config, bad configuration.')
        return 

    def _connection_failed(self, link_uri, msg):
        """
        Callback when connection initial connection fails (i.e no Crazyflie
        at the speficied address)
        """
        print('Connection to %s failed: %s' % (link_uri, msg))
        return 

    def _connection_lost(self, link_uri, msg):
        """
        Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)
        """
        print('Connection to %s lost: %s' % (link_uri, msg))
        return 

    def _disconnected(self, URI):
        self.view_mission_results()
        try:
            self.position_commander.land()
        except:
            pass
        print('Disconnected')
        return 

    def _log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))
        return

def get_required_points(detected_border_pad, is_hover= True):
    if is_hover:
        left_pad_border = detected_border_pad[0]
        left_pad_points = np.array(left_pad_border).reshape(-1,2)
        left_pad_median = np.median(left_pad_points, axis=0)    

        right_pad_border = detected_border_pad[1]
        right_pad_points = np.array(right_pad_border).reshape(-1,2)
        right_pad_median = np.median(right_pad_points, axis=0)
        return (left_pad_median[0] + right_pad_median[0])/2,  (left_pad_median[1] + right_pad_median[1])/2 

    else:
        left_pad_border = detected_border_pad[0]
        left_pad_points = np.array(left_pad_border).reshape(-1,2)
        left_pad_median = np.median(left_pad_points, axis=0)    

        right_pad_border = detected_border_pad[1]
        right_pad_points = np.array(right_pad_border).reshape(-1,2)
        right_pad_median = np.median(right_pad_points, axis=0)

        front_pad_border = detected_border_pad[2]
        front_pad_points = np.array(front_pad_border).reshape(-1,2)
        front_pad_median = np.median(front_pad_points, axis=0)

     
        return (left_pad_median[0] + front_pad_median[0] + right_pad_median[0])/3,  (left_pad_median[1] + right_pad_median[1] + front_pad_median[1])/3

def get_return_waypoints(x_values, y_values, home_position):
    waypoints: List[np.ndarray] = []
    for x,y in zip(x_values, y_values):
        waypoints.append(np.array([x-home_position[0],y-home_position[1]]))
    return waypoints

def get_neighbor_sensors(trigged_direction):
    direction_map = ["front", "left","back", "right"]
    position = direction_map.index(trigged_direction)
    if position==3:
        A = direction_map[0]
        B = direction_map[2]
    elif position == 0:
        A = direction_map[1]
        B = direction_map[3]
    else:
        A = direction_map[position+1]
        B = direction_map[position-1]
    return A, B

def get_optimal_forward_path(measurements, max_view = 1800):
    max_sensor_value = 0
    best_position = np.array([0.0,0.0])
    for sensor_value,position in measurements:
        if sensor_value>max_sensor_value:
            max_sensor_value = sensor_value
            best_position = position
    return best_position, max_sensor_value>max_view

def best_know_home_position(detected_border_pad):
    if (len(detected_border_pad[0])>2) and (len(detected_border_pad[1])>2):
        left_pad_border = detected_border_pad[0]
        left_pad_points = np.array(left_pad_border).reshape(-1,2)
        left_pad_median = np.median(left_pad_points, axis=0)    

        right_pad_border = detected_border_pad[1]
        right_pad_points = np.array(right_pad_border).reshape(-1,2)
        right_pad_median = np.median(right_pad_points, axis=0)
        return (left_pad_median[0] + right_pad_median[0])/2,  (left_pad_median[1] + right_pad_median[1])/2 
    elif (len(detected_border_pad[0])>2):
        left_pad_border = detected_border_pad[0]
        left_pad_points = np.array(left_pad_border).reshape(-1,2)
        left_pad_median = np.median(left_pad_points, axis=0)    
        
        return left_pad_median[0],left_pad_median[1]+0.25

    elif ((len(detected_border_pad[1])>2)):
        right_pad_border = detected_border_pad[1]
        right_pad_points = np.array(right_pad_border).reshape(-1,2)
        right_pad_median = np.median(right_pad_points, axis=0)
        
        return right_pad_median[0],right_pad_median[1]-0.25
    else:
        return 0.0,0.0

def get_average_sensor_measurements(sensor_measurements):
    data_frame = pd.DataFrame(sensor_measurements)
    sensor_measurement = dict(data_frame.mean())
    return sensor_measurement
