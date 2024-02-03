import numpy as np
import cv2



def KalmanFilter(measurements):
    '''
        # Assume we have a series of measurements (x, y)
        measurements = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])  # Example measurements
    '''
    # Example usage
    dt = 1.0  # Time step
    process_noise_std = 0.2  # Process noise standard deviation
    measurement_noise_std = 0.5  # Measurement noise standard deviation

    # kf = create_kalman_filter(dt, process_noise_std, measurement_noise_std)

    # Create a Kalman Filter instance
    kf = cv2.KalmanFilter(4, 2)

    # State Transition matrix
    kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                    [0, 1, 0, dt],
                                    [0, 0, 1,  0],
                                    [0, 0, 0,  1]], np.float32)

    # Measurement matrix
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], np.float32)

    # Process Noise Covariance
    kf.processNoiseCov = np.array([[1, 0, 0,    0],
                                    [0, 1, 0,    0],
                                    [0, 0, 1,    0],
                                    [0, 0, 0,    1]], np.float32) * process_noise_std**2

    # Measurement Noise Covariance
    kf.measurementNoiseCov = np.array([[1, 0],
                                        [0, 1]], np.float32) * measurement_noise_std**2



    # for measurement in measurements:
    #     # Update Kalman Filter with new measurement
    #     kf.correct(np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]))

    #     # Predict next state
    #     prediction = kf.predict()
    #     print("Predicted State:", prediction.flatten())

    # return prediction

    denoised_trajectory = []

    for measurement in measurements:
        # Update Kalman Filter with new measurement
        kf.correct(np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]]))

        # Predict next state
        prediction = kf.predict()
        denoised_trajectory.append(prediction.flatten())

        # print("Predicted State:", prediction.flatten())

    # denoised_trajectory now holds the denoised sequence of positions
    denoised_trajectory = np.array(denoised_trajectory)
    # print("\nDenoised Trajectory:\n", denoised_trajectory[:,:2])

    future_predictions = []

    for _ in range(100):
        # Predict next state without a new measurement
        future_state = kf.predict()
        future_predictions.append(future_state.flatten())

    # Convert the list of predictions to a numpy array for easier handling
    future_predictions = np.array(future_predictions)

    return denoised_trajectory[:,:2], future_predictions[:,:2]



if __name__ == "__main__":
    measurements = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])  # Example measurements
    KalmanFilter(measurements)
