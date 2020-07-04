import numpy as np
import matplotlib.pyplot as plt

# Load the training dataset
train_features = np.load("train_features.npy")
train_labels = np.load("train_labels.npy").astype("int8")

n_train = train_labels.shape[0]

def visualize_digit(features, label):
    # Digits are stored as a vector of 400 pixel values. Here we
    # reshape it to a 20x20 image so we can display it.
    plt.imshow(features.reshape(20, 20), cmap="binary")
    plt.xlabel("Digit with label " + str(label))
    plt.show()


# Question a) visualize 3 0's and 3 1's
zeros_cnt = 0
ones_cnt  = 0

for i in range(n_train):
    if train_labels[i] == 0 and zeros_cnt != 3:
        zeros_cnt += 1
        visualize_digit(train_features[i], train_labels[i])
    if train_labels[i] == 1 and ones_cnt != 3:
        ones_cnt += 1
        visualize_digit(train_features[i], train_labels[i])
    
    if zeros_cnt == 3 and zeros_cnt == 3:
        break


##### Linear regression

### Question b + c)  X = train_features,  y = 2 * train_labels - 1 = -1 or +1

print('\n----- Classification Task 1 (-1/+1) -----\n')
print('Number of training set examples: ' + str(n_train))
X = train_features
y = np.array([(2 * train_label - 1) for train_label in train_labels])   # either -1 (handwritten 0) or +1 (handwritten 1)

# find the optimal weights vector
w1 = np.linalg.inv(np.dot(X.T, X)) # (X^T * X)^-1
w2 = np.dot(w1, X.T)               # (X^T * X)^-1 * X^T
w3 = np.dot(w2, y)                 # (X^T * X)^-1 * X^T * y

print('Some of the weights are: ' + str(w3[:20]))  # weight vector
r = np.dot(X, w3) - y
print('Residual Error: ' + str(np.linalg.norm(r, 2))) # residual error 

correct_pred = 0
# check prediction accuracy on training set
for i in range(n_train):
    if w3.dot(X[i].T) <= 0:    # predict '0'
        predict = 0
    else:
        predict = 1     # predict '1'

    if predict == train_labels[i]:
        correct_pred += 1

print('Training-Set Accuracy: ' + str(correct_pred / n_train))


# Load the test dataset
# It is good practice to do this after the training has been
# completed to make sure that no training happens on the test
# set!
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy").astype("int8")

n_test = test_labels.shape[0]

correct_pred = 0
# check prediction accuracy on test set
for i in range(n_test):
    if w3.dot(test_features[i].T) <= 0:    # predict '0'
        predict = 0
    else:
        predict = 1     # predict '1'

    if predict == test_labels[i]:
        correct_pred += 1


print('\nNumber of test set examples: ' + str(n_test))
print('Test-Set Accuracy: ' + str(correct_pred / n_test))


### Question e)  X = train_features,  y = train_labels

print('\n----- Classification Task 2 (0/1) -----')
y = train_labels   # either 0 (for class 0) or 1 (for class 1)

# find the optimal weights vector
w1 = np.linalg.inv(np.dot(X.T, X)) # (X^T * X)^-1
w2 = np.dot(w1, X.T)               # (X^T * X)^-1 * X^T
w3 = np.dot(w2, y)                 # (X^T * X)^-1 * X^T * y

print('\nSome of the weights are: ' + str(w3[:20]))  # weight vector
r = np.dot(X, w3) - y
print('Residual Error: ' + str(np.linalg.norm(r, 2))) # residual error

correct_pred = 0
# check prediction accuracy on training set
for i in range(n_train):
    if w3.dot(X[i].T) <= 0:    # predict '0'
        predict = 0
    else:
        predict = 1     # predict '1'

    if predict == train_labels[i]:
        correct_pred += 1

print('Training-Set Accuracy: ' + str(correct_pred / n_train))


correct_pred = 0
# check prediction accuracy on training set
for i in range(n_test):
    if w3.dot(test_features[i].T) <= 0:    # predict '0'
        predict = 0
    else:
        predict = 1     # predict '1'

    if predict == test_labels[i]:
        correct_pred += 1


print('Test-Set Accuracy: ' + str(correct_pred / n_test))


### Question e)   X = train_features with bias column,  y = 2 * train_labels - 1 = -1 or +1
print('\n----- Classification Task 3 (X\', -1/+1) -----\n')
X_new = np.c_[X, np.ones(n_train)]  # add an extra bias column
y = np.array([(2 * train_label - 1) for train_label in train_labels])   # either -1 (0) or +1 (1)

# find the optimal weights vector
w1 = np.linalg.inv(np.dot(X_new.T, X_new)) # (X^T * X)^-1
w2 = np.dot(w1, X_new.T)               # (X^T * X)^-1 * X^T
w3 = np.dot(w2, y)                 # (X^T * X)^-1 * X^T * y

print('Some of the weights are: ' + str(w3[:20]))  # weight vector
r = np.dot(X_new, w3) - y
print('Residual Error: ' + str(np.linalg.norm(r, 2))) # residual error

correct_pred = 0
# check prediction accuracy on training set
for i in range(n_train):
    if w3.dot(X_new[i].T) <= 0:    # predict '0'
        predict = 0
    else:
        predict = 1     # predict '1'

    if predict == train_labels[i]:
        correct_pred += 1

print('Training-Set Accuracy: ' + str(correct_pred / n_train))

test_features = np.c_[test_features, np.ones(n_test)]

correct_pred = 0
# check prediction accuracy on training set
for i in range(n_test):
    if w3.dot(test_features[i].T) <= 0:    # predict '0'
        predict = 0
    else:
        predict = 1     # predict '1'

    if predict == test_labels[i]:
        correct_pred += 1


print('Test-Set Accuracy: ' + str(correct_pred / n_test))


### X = train_features with bias column,  y = train_labels
print('\n----- Classification Task 4 (X\', 0/1) -----\n')
y = train_labels   # either 0 (for class 0) or 1 (for class 1)

# find the optimal weights vector
w1 = np.linalg.inv(np.dot(X_new.T, X_new)) # (X^T * X)^-1
w2 = np.dot(w1, X_new.T)               # (X^T * X)^-1 * X^T
w3 = np.dot(w2, y)                 # (X^T * X)^-1 * X^T * y

print('Some of the weights are: ' + str(w3[:20]))  # weight vector
r = np.dot(X_new, w3) - y
print('Residual Error: ' + str(np.linalg.norm(r, 2))) # residual error

correct_pred = 0
# check prediction accuracy on training set
for i in range(n_train):
    if w3.dot(X_new[i].T) <= 0:    # predict '0'
        predict = 0
    else:
        predict = 1     # predict '1'

    if predict == train_labels[i]:
        correct_pred += 1

print('Training-Set Accuracy: ' + str(correct_pred / n_train))


correct_pred = 0
# check prediction accuracy on test set
for i in range(n_test):
    if w3.dot(test_features[i].T) <= 0:    # predict '0'
        predict = 0
    else:
        predict = 1     # predict '1'

    if predict == test_labels[i]:
        correct_pred += 1


print('Test-Set Accuracy: ' + str(correct_pred / n_test))

###############################################################################
