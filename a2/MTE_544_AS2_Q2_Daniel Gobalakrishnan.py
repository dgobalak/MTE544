"""
Student Name & Last Name: Daniel Gobalakrishnan
Origianl Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q2_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import random
import pandas as pd
def plot_ellipse(Q, b, ax):
    eigvals, eigvecs = np.linalg.eigh(Q)

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse_param = unit_circle / np.sqrt(eigvals[:, np.newaxis])
    ellipse_points = eigvecs @ ellipse_param + b[:, np.newaxis]

    ax.plot(ellipse_points[0, :], ellipse_points[1, :], 'b-', label='Fitted Ellipse')
    ax.plot(b[0], b[1], 'ro', label='Center')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    ax.grid(True)

def visualize_data(p, ax, inliers, threshold):
    ax.scatter(p[:, 0], p[:, 1], color='red', alpha=0.5, label='Raw Measurements (Ellipse)')
    ax.scatter(inliers[:, 0], inliers[:, 1], color='purple', alpha=0.7, label='Inliers')

    for point in inliers:
        circle = plt.Circle(point, threshold, color='orange', fill=False, linestyle='--', alpha=0.7)
        ax.add_patch(circle)


def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def fit_ellipse_subset(points):
    # Given some number of points (you have to determined this), 
    # construct an ellipse that fits through those points.

    ##### ADD your code here : #####
    A = np.zeros((5, 5))
    b = -np.ones((5, 1))

    for i in range(5):
        x_i = points[i][0]
        y_i = points[i][1]
        A[i] = np.array([x_i**2, 2*x_i*y_i, y_i**2, -2*x_i, -2*y_i])

    # Solve for the ellipse
    ellipse = scipy.linalg.solve(A, b)
    
    A_val = ellipse[0].item()
    B_val = ellipse[1].item()
    C_val = ellipse[2].item()
    D_val = ellipse[3].item()
    E_val = ellipse[4].item()
    
    # Define the Q matrix (without alpha scaling)
    prescale_Q = np.array([
        [A_val, B_val],
        [B_val, C_val]
    ])

    # Define the Qb vector (without alpha scaling)
    prescale_Qb = np.array([[D_val], [E_val]])

    # Calculate alpha
    alpha_denom = (prescale_Qb.T @ np.linalg.inv(prescale_Q) @ prescale_Qb) - 1
    alpha = 1 / alpha_denom

    # # Scale Q and b by alpha
    Q = alpha * prescale_Q
    b = np.linalg.inv(Q) @ (alpha * prescale_Qb)
    b = b.flatten()
            
    # ##### END #####
    return Q, b

def ransac_ellipse(data, num_iterations=1000, threshold=0.2):
    inliers = []
    # Given the data sets, perform RANSAC to find the best Q and b as well as the inliers
    # Hint: You should use fit_ellipse_subset 
    # Hint: in some case, the Q matrix might not be positive defintie, use is_positive_definite to check.

    ##### ADD your code here : #####
    best_inliers = np.array([])
    best_Q = None
    best_b = None

    for _ in range(num_iterations):
        # Randomly select a subset of 5 points
        subset = data[np.random.choice(data.shape[0], 5, replace=False)]

        # Fit ellipse to subset
        curr_Q, curr_b = fit_ellipse_subset(subset)

        # Check if Q is positive definite
        if not is_positive_definite(curr_Q):
            continue

        # Find inliers
        curr_inliers = []
        for point in data:
            diff = point - curr_b
            distance = np.sqrt(diff.T @ curr_Q @ diff) - 1
            if abs(distance) <= threshold:
                curr_inliers.append(point)

        # Update best fit if this model has the most inliers
        if len(curr_inliers) > len(best_inliers):
            best_inliers = curr_inliers
            best_Q = curr_Q
            best_b = curr_b

    Q = best_Q
    b = best_b.flatten()
    inliers = np.array(best_inliers)

    ##### END #####
    return Q,b,inliers


if __name__ == "__main__":
    # Load the data from CSV file and select N random points
    N = 500
    threshold = 5
    num_iterations = 1000
    all_data = pd.read_csv('data_x_y.csv').to_numpy()
    dataset = all_data[np.random.choice(all_data.shape[0], N, replace=False), :]
    # dataset is p
    
    Q, b_est, inliers = ransac_ellipse(dataset, num_iterations=num_iterations, threshold=threshold)
    
    # Plot the raw measurements and fitted ellipse
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    visualize_data(dataset, ax1, inliers, threshold=threshold)
    plot_ellipse(Q, b_est, ax1)
    ax1.set_title("RANSAC Ellipse Fitting with Threshold Visualization")

    plt.show()
