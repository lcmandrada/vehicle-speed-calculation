# Vehicle Speed Calculation

**Legacy Project**

Optimization of Vehicle Speed Calculation on Raspberry Pi Using Sparse Random Projection  
A Thesis Report

In order to weaken the barrier of implementation of speed limit enforcement in the country, ways to improve vehicle speed calculation such as image processing can be explored. It was observed that existing prototypes suffer with low effective frame rate, the average rate at which the system processes video frames, considering it unfit for real-time setup. In this study, a vehicle speed calculation system was developed on Raspberry Pi with Gaussian Mixture Model for vehicle detection and Kalman Filter for vehicle tracking on OpenCV, and was optimized with Sparse Random Projection on scikit-learn by projecting the video to a low-dimensional subspace. The prototype was tested and analyzed in performance, in terms of effective frame rate, and in accuracy, in terms of vehicle speed, by applying paired t-test and linear regression analysis to prove if the optimization improved the performance and accuracy of vehicle speed calculation. It was found that Sparse Random Projection significantly improved the performance of the system at 7.08 fps and in effect improved accuracy with significant correlation between actual and calculated vehicle speed at an average absolute difference error of 0.76 kph. In contrary, vehicle speed calculation without optimization performed at 3.29 fps and 1.25 kph difference error.

# Note

The project is written for Raspberry Pi and uses a modified version of random_projection.py from sklearn.