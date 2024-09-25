import numpy as np
import matplotlib.pyplot as plt

# Define the system of equations (kinematic model)
def robot_kinematics(state, v, omega, dt):
    x, y, theta = state
    dxdt = v * np.cos(theta)
    dydt = v * np.sin(theta)
    dthetadt = omega
    # Update the state using Euler integration
    x_new = x + dxdt * dt
    y_new = y + dydt * dt
    theta_new = theta + dthetadt * dt
    return [x_new, y_new, theta_new]

# Simulation parameters
time_span = 30  # 30 seconds
dt = 0.1  # Time step of 0.1s
time_steps = int(time_span / dt)
time_array = np.arange(0, time_span + dt, dt)

# Define velocity profiles
def velocity_profile_1(t):
    return 1, 0  # [v, omega]

def velocity_profile_2(t):
    return 0, 0.3  # [v, omega]

def velocity_profile_3(t):
    return 1, 0.3  # [v, omega]

def velocity_profile_4(t):
    v = 1 + 0.1 * np.sin(t)
    omega = 0.2 + 0.5 * np.cos(t)
    return v, omega

# Simulate and store the results
def simulate_trajectory(profile_func, initial_state=[0, 0, 0]):
    # Initialize state and storage
    state = initial_state
    x_vals = [state[0]]
    y_vals = [state[1]]
    theta_vals = [state[2]]
    
    # Simulate over the given time span
    for i in range(time_steps):
        t = i * dt
        v, omega = profile_func(t)
        state = robot_kinematics(state, v, omega, dt)
        x_vals.append(state[0])
        y_vals.append(state[1])
        theta_vals.append(state[2])
    
    return x_vals, y_vals, theta_vals

# Plot results for each velocity profile
def plot_trajectory_and_variables(x_vals, y_vals, theta_vals, profile_label):
    # Create a figure for each profile
    plt.figure(figsize=(10, 10))

    # Plot the overhead 2D trajectory (x vs y)
    plt.subplot(4, 1, 1)
    plt.plot(x_vals, y_vals, label=profile_label)
    plt.title(f'2D Trajectory: {profile_label}')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.legend()

    # Plot x(t)
    plt.subplot(4, 1, 2)
    plt.plot(time_array, x_vals, label='x(t)')
    plt.ylabel('x [m]')
    plt.grid()

    # Plot y(t)
    plt.subplot(4, 1, 3)
    plt.plot(time_array, y_vals, label='y(t)')
    plt.ylabel('y [m]')
    plt.grid()

    # Plot θ(t)
    plt.subplot(4, 1, 4)
    plt.plot(time_array, theta_vals, label='θ(t)')
    plt.xlabel('Time [s]')
    plt.ylabel('θ [rad]')
    plt.grid()

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

# Velocity profile 1: v=1, ω=0
x_vals, y_vals, theta_vals = simulate_trajectory(velocity_profile_1)
plot_trajectory_and_variables(x_vals, y_vals, theta_vals, 'v=1, ω=0')

# Velocity profile 2: v=0, ω=0.3
x_vals, y_vals, theta_vals = simulate_trajectory(velocity_profile_2)
plot_trajectory_and_variables(x_vals, y_vals, theta_vals, 'v=0, ω=0.3')

# Velocity profile 3: v=1, ω=0.3
x_vals, y_vals, theta_vals = simulate_trajectory(velocity_profile_3)
plot_trajectory_and_variables(x_vals, y_vals, theta_vals, 'v=1, ω=0.3')

# Velocity profile 4: v(t) = 1 + 0.1*sin(t), ω(t) = 0.2 + 0.5*cos(t)
x_vals, y_vals, theta_vals = simulate_trajectory(velocity_profile_4)
plot_trajectory_and_variables(x_vals, y_vals, theta_vals, 'v(t) = 1 + 0.1*sin(t), ω(t) = 0.2 + 0.5*cos(t)')
