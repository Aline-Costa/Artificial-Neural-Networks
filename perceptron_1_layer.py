import numpy as np

def stepFunction(summation):

    if (summation >= 1):
        return 1
    return 0


def classify(input):

    s = input.dot(weights)  # dot product  
    return stepFunction(s)


def fit():

    total_error = 1
    while (total_error != 0):  # while as the accuracy is not 100%, continue...
        total_error = 0
        for i in range(len(outputs_real)):
            classification = classify(np.asarray(inputs[i]))
            error = outputs_real[i] - classification
            total_error += error
            for j in range(len(weights)):
                weights[j] = weights[j] + (learning_rate * inputs[i][j] * error)  # update weights
        print("Updated weight: " + str(weights[j]))
        print("Total errors: " + str(total_error))



# AND Logic
inputs = np.array([[0,0],[0,1], [1,0], [1,1]])    
outputs_real = np.array([0,0,0,1])

# OR Logic
#inputs  = np.array([[0,0],[0,1], [1,0], [1,1]])
#outputs_real  = np.array([0,1,1,1])

weights = np.array([0.0, 0.0])    # starting weight
learning_rate = 0.1
fit()
