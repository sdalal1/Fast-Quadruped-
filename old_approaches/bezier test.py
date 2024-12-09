import numpy as np
import matplotlib.pyplot as plt

# Define 6 control points for the Bezier curve
# control_points = np.array([[-0.2, 0.5],
#                            [-0.28,0.5],
#                            [-0.3,0.361], 
#                            [-0.3, 0.361], 
#                            [-0.3,0.361],
#                            [0.0,0.361],
#                            [0,0.361],
#                            [0,0.3214],
#                            [0.3032,0.3214],
#                            [0.3032,0.3214], 
#                            [0.2826,0.5],
#                            [0.2,0.5]])
# control_points = np.array([[-0.05, 0.2],
#                            [-0.13,0.2],
#                            [-0.15,0.061], 
#                            [-0.15, 0.061], 
#                            [-0.15,0.061],
#                            [0.0,0.061],
#                            [0,0.061],
#                            [0,0.0214],
#                            [0.1532,0.0214],
#                            [0.1532,0.0214], 
#                            [0.1326,0.2],
#                            [0.05,0.2]])

# control_points = np.array([ [ 0.   ,   0.    ], 
#                             [-0.03  ,  0.    ],
#                             [-0.05  ,  -0.059 ],
#                             [-0.05  , -0.059 ],
#                             [-0.05  ,  -0.059 ],
#                             [ 0.0 ,  -0.059 ],
#                             [ 0.0 ,  -0.059 ],
#                             [ 0.0 ,  -0.09],
#                             [ 0.15, -0.09],
#                             [ 0.15, -0.09],
#                             [ 0.13,  0.    ],
#                             [ 0.05  ,  0.    ]])
        
control_points = np.array([[-0.49 , -0.28 ],
                            [-0.52 , -0.28 ],
                            [-0.54 , -0.339],
                            [-0.54 , -0.339],
                            [-0.54 , -0.339],
                            [-0.49 , -0.339],
                            [-0.49 , -0.339],
                            [-0.49 , -0.37 ],
                            [-0.34 , -0.37 ],
                            [-0.34 , -0.37 ],
                            [-0.36 , -0.28 ],
                            [-0.44 , -0.28 ]])

dog_points = np.array([(0.49, 0.27999999999999997), (0.36, 0.45999999999999996), (0.5, 0.7), (0.0, 0.7), (-0.5, 0.7), (-0.33999999999999997, 0.44999999999999996), (-0.62, 0.30999999999999994)])

b_thigh_theta = -np.pi/4
b_calf_theta = -np.pi/4
f_thigh_theta = -np.pi/4
f_calf_theta = -np.pi/4

l_b_thigh = 0.145
l_b_calf = 0.15

# 
l_f_thigh = 0.133
l_f_calf = 0.106
l_torso = 1

# dog_points = np.array([ (l_torso/2 - (l_f_thigh * np.cos(f_thigh_theta))+(l_f_calf * np.cos(f_calf_theta)), l_f_thigh * np.sin(f_thigh_theta) + l_f_calf * np.sin(f_calf_theta)),
#                         (l_torso/2 - (l_f_thigh * np.cos(f_thigh_theta)), l_f_thigh * np.sin(f_thigh_theta)),
#                         (l_torso/2, 0), 
#                         (-l_torso/2, 0), 
#                         (-l_torso/2 + (l_b_thigh * np.cos(b_thigh_theta)), l_b_thigh * np.sin(b_thigh_theta)),
#                         (-l_torso/2 + (l_b_thigh * np.cos(b_thigh_theta) - l_b_calf * np.cos(b_calf_theta)), l_b_thigh * np.sin(b_thigh_theta) + l_b_calf * np.sin(b_calf_theta)),
#                         ])


dog_points -= dog_points[0] 
# control_points -= control_points[0]
# print(control_points)
# control_points1 =  control_points - control_points[-1]

# control_points = np.array([(0.49, 0.27999999999999997),])
# Define parameter t (0 <= t <= 1) for interpolation
t = np.linspace(0, 1, 100)

# Calculate Bezier curve points using De Casteljau algorithm
def bezier_curve(t, control_points):
    n = len(control_points) - 1
    points = np.copy(control_points)
    curve_points = np.zeros((len(t), 2))  # Initialize array to store curve points
    for i, ti in enumerate(t):
        b = np.zeros((n + 1, 2))  # Initialize array to store intermediate points
        b[:, :] = points  # Set first row of b to control points
        for r in range(1, n + 1):
            for j in range(n - r + 1):
                b[j] = (1 - ti) * b[j] + ti * b[j + 1]
        curve_points[i] = b[0]
    return curve_points

def stance(x, delta, y):
    res = []
    l = x[0] - x[-1]
    if l == 0:
        for i in range(len(x)):
            y_c = -1*delta*np.sin(np.pi*i/len(x))+y
            res.append(y_c)
    else:
        for i in range(len(x)):
            y_c = -1*delta*np.sin(np.pi*(x[i]-x[0])/l)+y
            res.append(y_c)
    return res

# Generate trajectory points along the Bezier curve
trajectory = bezier_curve(t, control_points)

sin_x = np.linspace(control_points[-1][0], control_points[0][0], 100)
sin_y = stance(sin_x, 0.01, control_points[0][1])

mv_x = np.concatenate((sin_x, trajectory[:, 0]))
mv_y = np.concatenate((sin_y, trajectory[:, 1]))

# traj1 = bezier_curve(t, control_points1)



plt.plot(trajectory[:, 0], -trajectory[:, 1], 'bo-', label='Swing')
plt.plot(control_points[:, 0], -control_points[:, 1], 'ro--', label='Control Points')
# plt.plot(traj1[:, 0], -traj1[:, 1], 'g-', label='Swing1')
plt.plot(mv_x, -mv_y, 'b-', label='Swinging')
# plt.plot(sin_x, sin_y, 'r-', label='Stance')

plt.plot(dog_points[:, 0], dog_points[:, 1], 'go--', label='Dog Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()


# Plot Bezier curve and control points
# plt.plot(control_points[:, 0], -control_points[:, 1], 'ro--', label='Control Points')
# plt.plot(trajectory[:, 0], -trajectory[:, 1], 'b-', label='Bezier Curve')
# plt.plot(dog_points[:, 0], dog_points[:, 1], 'go--', label='Dog Points')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Define 6 control points for the Bezier curve
# control_points = np.array([[0.5,0],[0.2,0.2],[0.5,0.4],[0,0.4],[-0.5,0.4],[-0.2,0.2],[-0.5,0]])

# # Define parameter t (0 <= t <= 1) for interpolation
# t = np.linspace(0, 1, 100)

# # Function to update all control points dynamically
# def update_control_points(t):
#     # Example: Move all points along a sine wave in the x-direction
#     new_x = 0.5 * np.sin(2 * np.pi * t)
#     # Update the x-coordinates of all control points
#     control_points[:, 0] = new_x

# # Calculate Bezier curve points using De Casteljau algorithm
# def bezier_curve(t, control_points):
#     n = len(control_points) - 1
#     points = np.copy(control_points)
#     curve_points = np.zeros((len(t), 2))  # Initialize array to store curve points
#     for i, ti in enumerate(t):
#         update_control_points(ti)  # Update all control points based on t
#         b = np.zeros((n + 1, 2))  # Initialize array to store intermediate points
#         b[:, :] = points  # Set first row of b to control points
#         for r in range(1, n + 1):
#             for j in range(n - r + 1):
#                 b[j] = (1 - ti) * b[j] + ti * b[j + 1]
#         curve_points[i] = b[0]
#     return curve_points

# # Initialize plot
# fig, ax = plt.subplots()
# line, = ax.plot([], [], 'b-', label='Bezier Curve')
# points, = ax.plot([], [], 'ro--', label='Control Points')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Bezier Curve Animation with 6 Control Points (All Points Moving)')
# ax.legend()
# ax.grid(True)
# ax.axis('equal')

# # Function to initialize the plot
# def init():
#     line.set_data([], [])
#     points.set_data([], [])
#     return line, points

# # Function to update the plot for each frame of the animation
# def update(frame):
#     trajectory = bezier_curve(t, control_points)
#     line.set_data(trajectory[:, 0], trajectory[:, 1])
#     points.set_data(control_points[:, 0], control_points[:, 1])
#     return line, points

# # Create animation
# ani = FuncAnimation(fig, update, frames=np.linspace(0, 1, 200), init_func=init, blit=True)

# # Show animation
# plt.show()

# import sympy as sp
# import numpy as np

# x,y,x1,l1,l2 = sp.symbols('x y x1 l1 l2')
# theta1, theta2 = sp.symbols('theta1 theta2')



# f1 = sp.simplify(x1-x-l1*sp.cos(theta1)-l2*sp.cos(theta2-theta1))
# f2 = sp.simplify(y-l1*sp.sin(theta1)-l2*sp.sin(theta2-theta1))

# # solve for theta2 and theta1 where theta2 and theta1 can depend on each other
# #

# sol = sp.solve([f1,f2],(theta1,theta2))
# print(sol)



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Define control points
# control_points = np.array([[0, 0], [1, 1], [2, -1], [3, 0]])

# # Bezier curve function
# def bezier(t):
#     n = len(control_points) - 1
#     return sum([((1 - t) ** (n - i)) * (t ** i) * control_points[i] * np.math.comb(n, i) for i in range(n + 1)])

# # Animation function
# def animate(frame):
#     plt.cla()
#     t = frame / 100.0
#     # Plot control points
#     plt.plot(control_points[:,0], control_points[:,1], 'ro')
#     # Plot bezier curve
#     curve_points = np.array([bezier(t_) for t_ in np.linspace(0, t, 100)])
#     plt.plot(curve_points[:,0], curve_points[:,1])
#     plt.xlim(-1, 4)
#     plt.ylim(-2, 2)
#     plt.title('Bezier Curve Animation')

# # Create animation
# fig = plt.figure()
# ani = FuncAnimation(fig, animate, frames=range(100), repeat=True)
# plt.show()
