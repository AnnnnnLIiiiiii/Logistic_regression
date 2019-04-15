import numpy as np
import h5py



def load_dataset(database_path):
    # open dataset
    dataset_db = h5py.File(database_path, "r")

    datasets = {}
    for dataset in ["train", "dev", "test"]:
        # load the train set feautres (picuture)
        datasets[dataset] = {'X': np.array(dataset_db[dataset + "_img"][:]),  # dataset features
                             'Y': np.array(dataset_db[dataset + "_labels"][:])  # dataset labels
                            }
    return datasets

def organzie_data(datasets):
    car_data_set = load_dataset("./car_dataset.hdf5")

    raw_car_train_x = car_data_set["train"]["X"]
    raw_car_train_y = car_data_set["train"]["Y"]
    raw_car_dev_x = car_data_set["dev"]["X"]
    raw_car_dev_y = car_data_set["dev"]["Y"]
    raw_car_test_x = car_data_set["test"]["X"]
    raw_car_test_y = car_data_set["test"]["Y"]

    raw_car_train_y = np.array([0 if y == 1 else 1 for y in raw_car_train_y])
    raw_car_test_y = np.array([0 if y == 1 else 1 for y in raw_car_test_y])
    raw_car_dev_y = np.array([1 if y == 0 else 0 for y in raw_car_dev_y])

    car_train_x = raw_car_train_x.reshape(raw_car_train_x.shape[0], -1).T / 255.
    car_dev_x = raw_car_dev_x.reshape(raw_car_dev_x.shape[0], -1).T / 255.
    car_test_x = raw_car_test_x.reshape(raw_car_test_x.shape[0], -1).T / 255.

    car_train_y = raw_car_train_y.reshape(1, -1)
    car_dev_y = raw_car_dev_y.reshape(1, -1)
    car_test_y = raw_car_test_y.reshape(1, -1)

    raw_X_dict = {"raw_car_train_x": raw_car_train_x,
                  "raw_car_dev_x": raw_car_dev_x,
                  "raw_car_test_x": raw_car_test_x
                  }
    data_dict = {"car_train_x": car_train_x,
                 "car_train_y": car_train_y,
                 "car_dev_x": car_dev_x,
                 "car_dev_y": car_dev_y,
                 "car_test_x": car_test_x,
                 "car_test_y": car_test_y
                 }

    return raw_X_dict, data_dict


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s


def lg_initialize_with_zeros(dim):
    w = np.zeros((dim, 1), float)
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def lg_f_prop(w, b, X, Y):
    m = X.shape[1]  # compute the number of trainig data
    # FORWARD PROPAGATION (FROM X TO COST)
    z = np.dot(w.T, X) + b
    A = sigmoid(z)  # compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return A, cost


def lg_back_prop(X, Y, A):
    # From backward propagation, we have
    # dL/dw1 = dL/dz * x1 -> dw1 = dz1 * x1 = (a1 - y1) * x1
    # where x1 is the first attribute of a data point.
    # We want dw to have same shape of w, therefore

    m = X.shape[1]
    dz = A - Y
    dw = (1 / m) * np.dot(X, dz.T)  # n * 1
    db = (1 / m) * np.sum(dz)

    return dw, db


def lg_optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    dbs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        A, cost = lg_f_prop(w, b, X, Y)
        dw, db = lg_back_prop(X, Y, A)
        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            dbs.append(db)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs, dbs


def lg_predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    z = np.dot(w.T, X) + b
    A = sigmoid(z)

    # Convert probabilities A[0,i] to actual predictions p[0,i]
    Y_prediction = np.array([1 if a > 0.5 else 0 for a in A[0]])
    Y_prediction = Y_prediction.reshape((1, -1))

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def lg_model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # initialize parameters with zeros (≈ 1 line of code)
    w, b = lg_initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs, dbs = lg_optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = lg_predict(w, b, X_test)
    Y_prediction_train = lg_predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "b_grad": dbs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d