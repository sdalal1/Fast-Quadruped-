from math import *
import numpy as np

def theta_front_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    # x = x + (torso_length/2)
    theta2 = np.pi - acos((x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc))
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    
    theta2 -= real_calf_angle
    theta1 -= real_thigh_angle
    return theta1, theta2

def theta_front_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    # x = x + (torso_length/2)
    print(x, y, lt, lc, torso_length) 
    theta2 = np.pi + acos((x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc))
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    theta2 -= real_calf_angle
    theta1 -= real_thigh_angle
    
    return theta1, theta2

def theta_back_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    # x = x - (torso_length/2)
    theta2 = np.pi - acos((x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc))
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    theta2-= real_calf_angle
    theta1-= real_thigh_angle
    return theta1, theta2

def theta_back_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    # x = x - (torso_length/2)
    theta2 = np.pi + acos((x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc))
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    theta2-= real_calf_angle
    theta1-= real_thigh_angle
    return theta1, theta2

def pick_case_front(x, y, lt, lc, torso_length, current_theta1, current_theta2, threshold, real_calf_angle, real_thigh_angle):
    # print(x, y, lt, lc, torso_length, current_theta1, current_theta2, threshold ) 
    # x = x - (torso_length/2)
    t11, t21 = theta_front_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    t12, t22 = theta_front_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    if np.sqrt(t11**2 + t21**2) < np.sqrt(t12**2 + t22**2):
        return t11, t21
    else:
        return t12, t22
    
def pick_case_back(x, y, lt, lc, torso_length, current_theta1, current_theta2, threshold, real_calf_angle, real_thigh_angle):
    # x = x + (torso_length/2)
    t11, t21 = theta_back_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    t12, t22 = theta_back_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    if np.sqrt(t11**2 + t21**2) < np.sqrt(t12**2 + t22**2):
        return t11, t21
    else:
        return t12, t22