import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import glob
from sklearn.model_selection import train_test_split
from Logistic_regression import *

# Load and organize data from files
car_data_set = load_dataset("./car_dataset.hdf5")
raw_dict, data_dict = organzie_data(car_data_set)

# Randomly show training and testing data
index = np.random.randint(0, 100)
fig=plt.figure(figsize=(11, 11))
fig.add_subplot(1, 2, 1, title = "%d in train_X, lable train_Y is %d" %(index, data_dict["car_train_y"][0][index]))
plt.imshow(raw_dict["raw_car_train_x"][index])
fig.add_subplot(1, 2, 2, title = "%d in test_X, lable test_Y is %d" %(index, data_dict["car_test_y"][0][index]))
plt.imshow(raw_dict["raw_car_test_x"][index])
print("press any key to continue")
plt.show()

# Training
num_iterations = int(input("Enter the desired iteration, the default value is 2000: "))
learning_rate = float(input("Enter the desired learning rate, the default value is 0.5: "))
d_lg = lg_model(data_dict["car_train_x"], data_dict["car_train_y"], data_dict["car_test_x"], data_dict["car_test_y"],
                num_iterations, learning_rate, print_cost=True)

# Plot the costs correspoidng to iteration
costs = np.squeeze(d_lg['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d_lg["learning_rate"]))
plt.show()

# Visualize the training result
predictions = lg_predict(d_lg["w"], d_lg["b"], data_dict["car_test_x"])
fig = plt.figure(figsize=(11, 11))
columns = 3
rows = 1
for i in range(1, columns*rows +1):
    index = np.random.randint(raw_dict["raw_car_test_x"].shape[1])
    img = raw_dict["raw_car_test_x"][index]
    fig.add_subplot(rows, columns, i, xlabel = "Fig %d" %index, title = "Lable is %d; Prediction is %d"
                    %(data_dict["car_test_y"][0,index], predictions[0, index]))
    plt.imshow(img)
    if data_dict["car_test_y"][0,index] == predictions[0, index]:
        print("For prediction of Fig. %d, the prediction is correct" %index)
    else:
        print("For prediction of Fig. %d, the prediction is wrong" %index)
plt.show()