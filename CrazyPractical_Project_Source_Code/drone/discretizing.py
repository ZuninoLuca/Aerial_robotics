import scipy.interpolate as interpolate
import numpy as np
from typing import List

def package_point(x_value,y_value):
    return np.array([round(x_value,2),round(y_value,2)])

def get_discretized_path(pt1:np.ndarray,pt2:np.ndarray, step_size: float = 0.05):
    discrete_line_x = interpolate.interp1d(x = [0,1], y = [pt1[0],pt2[0]])
    discrete_line_y = interpolate.interp1d(x = [0,1], y = [pt1[1],pt2[1]])

    total_points = int(np.linalg.norm(pt2-pt1)/step_size)

    x_values = discrete_line_x(np.linspace(0,1,num = total_points))
    y_values = discrete_line_y(np.linspace(0,1,num = total_points))
    
    points: List[np.ndarray] = []
    for x_value,y_value in zip(x_values,y_values):
        points.append(package_point(x_value,y_value))
    
    # disregard the first point
    points.pop(0)
    return points
    
def sweep_coverage_points(cruise_height, way_points: List[np.ndarray], home_x, home_y, expand_border = False):
    n = 0
    x_list = []
    y_list = []
    dir_flag = 0

    while ((n)*0.25*cruise_height <= 1.55):
        if (dir_flag == 0):
            if expand_border:
                y_list.append(3.6)
                y_list.append(3.6)
            else:
                y_list.append(3.2)
                y_list.append(3.2)
            dir_flag = 1
        else:
            if expand_border:
                y_list.append(-0.6)
                y_list.append(-0.6)
            else:
                y_list.append(0)
                y_list.append(0)
            dir_flag = 0
        x_list.append(n*0.25*cruise_height + 3.5)
        n += 1
        x_list.append(n*0.25*cruise_height + 3.5)
    y_list.pop()
    x_list.pop()
    while(len(y_list) != 0):
        way_points.append(np.array([x_list.pop(0) - home_x, y_list.pop(0) - home_y]))
    
    return way_points


def home_point_coverage(width, starting, height):
    path = []
    for i in range(int(height*10)+1):
        if i % 2 == 0:
            path.append(np.array([starting-i*0.1, -width/2]))
            path.append(np.array([starting-i*0.1, width/2]))
        else:
            path.append(np.array([starting-i*0.1, width/2]))
            path.append(np.array([starting-i*0.1, -width/2]))
        
    return path

