"""
Student Name & Last Name: Daniel Gobalakrishnan
Origianl Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q4_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

"""

import numpy as np

def decode_measurement(measurements):
    readings = []
    for measurement in measurements:
        readings.append(np.argmax(measurement) + 1)  # +1 for 1-based indexing
    return readings

def moodeng_behavior_update(state, A):
    # Given the current state, and the transition matrix A, 
    # randomly return the next state of Moo-Deng based on A
    ##### ADD your code here : #####
    row = A[state - 1]
    row = row / np.sum(row)
    next_state = np.random.choice([1, 2, 3], p=row)
    ##### END #####
    return next_state
def sensor_measurement(state, C):
    # Given a state, and the matrix C, 
    # randomly return the encoded measurement based on C
    # Note that : 
    # F -> np.array([1,0,0])
    # R -> np.array([0,1,0])
    # P -> np.array([0,0,1])
    ##### ADD your code here : #####
    row = C[state - 1]
    row = row / np.sum(row)
    measurement = np.random.choice([1, 2, 3], p=row)
    measurement = np.eye(3)[measurement - 1].T    
    ##### END #####
    return measurement

def sim_moodeng(initial_state=1,iteration = 20):
    # Given an initial state of Moo-Deng's whereabout and number of iteration,
    # simulate Moo-Deng's behavior and the designed state estimator
    
    belief = np.array([1/3, 1/3, 1/3])  # Initial belief for the Bayesian filter
    
    ##### ADD your code here : #####
    A = np.array([
        [0.6, 0.2, 0.2],
        [0.4, 0.4, 0.1],
        [0.0, 0.4, 0.7] 
    ])
    
    C = np.array([
        [0.8, 0.2, 0.05],
        [0.1, 0.7, 0.1],
        [0.1, 0.1, 0.85]
    ])
    ##### END #####

    states = []
    measurements = []
    estimated_states = []
    beliefs = []
    state = initial_state
    for i in range(iteration):
               
        ##### ADD your code here : #####
        next_state = moodeng_behavior_update(state, A)
        measurement = sensor_measurement(next_state, C)

        predicted_belief = A @ belief
        updated_belief = (C @ measurement) * predicted_belief
        
        # Normalize belief
        updated_belief = updated_belief / np.sum(updated_belief)
                
        estimated_state = np.argmax(updated_belief) + 1
        
        belief = updated_belief
        state = next_state

        ##### END #####
        states.append(next_state)
        measurements.append(measurement)
        beliefs.append(updated_belief)
        estimated_states.append(estimated_state)
        
    return states, measurements, estimated_states, beliefs

    
# Run the simulation

states, measurements, estimated, beliefs = sim_moodeng(initial_state=1, iteration=30)

readings = decode_measurement(measurements)

# Print results
print("True states:          ", states)
print("Sensor measurements:  ", readings)
print("Estimated states:     ", estimated)
print("Belief sequence:      ", np.array(beliefs))
