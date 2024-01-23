
import numpy as np




def oryginal_f(parameters):
    return parameters['x1'] ** 2 + parameters['x2'] ** 2 - 6 * parameters['x1'] - 4 * parameters['x2'] + 13

number_of_testing_values = 10
testingValues = {}

for i in range(0, number_of_testing_values):
    testingValues[i] = {}
    testingValues[i]['x1'] = np.random.randint(-10, 10)
    testingValues[i]['x2'] = np.random.randint(-10, 10)
    testingValues[i]['y'] = oryginal_f(testingValues[i])

def model_f(parameters):
    return parameters['x1'] ** 2 + parameters['x2'] ** 2 - parameters['a'] * parameters['x1'] - parameters['b'] * parameters['x2'] + parameters['c']

# for i in range(0, number_of_testing_values):
#     print('testing y: ', testingValues[i]['y'])
#     print('model y: ', model_f(dict(testingValues[i], **{'a': 6, 'b': 4, 'c': 13})))
#     print()

def mean_squared_error(parameters, model_function, data):
    sum = 0
    sum2 = 0
    varition = 0
    for i in range(0, len(data)):
        sum += (data[i]['y'] - model_function(dict(data[i], **parameters))) ** 2
        sum2 += np.abs(data[i]['y'] - model_function(dict(data[i], **parameters)))
    
    mean = sum / len(data)
    for i in range(0, len(data)):
        varition += (data[i]['y'] - mean) ** 2
    varition = varition / len(data)

    return mean

#print('mean_squared_error: ', mean_squared_error({'a': 6, 'b': 4, 'c': 13}))

def gradient_mean_squared_error(parameters, model_function, data):
    delta = 0.01
    updated_parameters = {}
    for key, value in parameters.items():
        updated_parameters[key] = value + delta
    nextValue = {key: mean_squared_error(dict(parameters, **{key: updated_parameters[key]}), model_function, data) for key, value in parameters.items()}
    gradient = {key: (nextValue[key] - mean_squared_error(parameters, model_function, data)) / delta for key, value in parameters.items()}
    return gradient

#print('gradient_mean_squared_error: ', gradient_mean_squared_error({'a': 8, 'b': 4, 'c': 13}))

def find_learning_rate(parameters, gradient, model_function, data):
    left = 0.0
    right = 1.0
    delta = 0.000000000001

    while right - left > delta:
        middle = (left + right) / 2.0
        checking_parameters_left = {key: parameters[key] - (left) * gradient[key] for key, value in parameters.items()}
        checking_parameters_right = {key: parameters[key] - (right) * gradient[key] for key, value in parameters.items()}
        left_value = mean_squared_error(checking_parameters_left, model_function, data)
        right_value = mean_squared_error(checking_parameters_right, model_function, data)
        # print('left', left, 'left_value: ', left_value, 'parameters: ', checking_parameters_left)
        # print('right', right, 'right_value: ', right_value, 'parameters: ', checking_parameters_right)
        if(any(value <= 0 for value in checking_parameters_left.values())):
            left = left + (middle - left) / 10.0
        elif(any(value <= 0 for value in checking_parameters_right.values())):
            right = right - (right - middle) / 10.0
        elif left_value < right_value:
            right = middle
        else:
            left = middle

        # if left_value < right_value:
        #     right = middle
        # else:
        #     left = middle

    # print('left: ', left)
    # print('right: ', right)
    # print('middle: ', (left + right) / 2.0)
    # print('left mean_squared_error: ', mean_squared_error({key: parameters[key] - (left) * gradient[key] for key, value in parameters.items()}, model_function, data))
    # print('right mean_squared_error: ', mean_squared_error({key: parameters[key] - (right) * gradient[key] for key, value in parameters.items()}, model_function, data))
    # print('mean_squared_error: ', mean_squared_error({key: parameters[key] - (left + right) / 2.0 * gradient[key] for key, value in parameters.items()}, model_function, data))
    # print()

    return (left + right) / 2.0



def find_parameters(start_parameters, model_function, data, iteration_limit = 1000):
    current_parameters = start_parameters
    iteration = 0
    while iteration < iteration_limit:
        iteration += 1
        print('Iteracja: ', iteration)
        print('current_parameters: ', current_parameters)
        print('mean_squared_error: ', mean_squared_error(current_parameters, model_function, data))
        print('gradient_mean_squared_error: ', gradient_mean_squared_error(current_parameters, model_function, data))
        print()
        gradient = gradient_mean_squared_error(current_parameters, model_function, data)
        learning_rate = find_learning_rate(current_parameters, gradient, model_function, data)
        current_parameters = {key: current_parameters[key] - learning_rate * gradient[key] for key, value in current_parameters.items()}
    
    return current_parameters

#print('find_parameters: ', find_parameters({'a': 8, 'b': 4, 'c': 13}, model_f, testingValues))



# def gradient_f(x1, x2):
#     gradient_x1 = 2 * x1 - 6
#     gradient_x2 = 2 * x2 - 4
#     return [gradient_x1, gradient_x2]

# def gradient_simple_f(x1, x2):
#     gradient_x1 = (f(x1 + 0.0001, x2) - f(x1, x2)) / 0.0001
#     gradient_x2 = (f(x1, x2 + 0.0001) - f(x1, x2)) / 0.0001
#     return [gradient_x1, gradient_x2]

# x1 = 0
# x2 = 0

# iteration = 0

# while iteration < 10:
#     iteration += 1
#     print('Iteracja: ', iteration)
#     print('x1: ', x1)
#     print('x2: ', x2)
#     print('f(x1, x2): ', f(x1, x2))
#     print('gradient_f(x1, x2): ', gradient_simple_f(x1, x2))
#     gradient = gradient_simple_f(x1, x2)
#     t = 1
#     prev_diff = 1000
#     next_diff = np.abs((f(x1 - t * gradient[0], x2 - t * gradient[1]) - f(x1 - t * gradient[0] + 0.0001, x2 - t * gradient[1] + 0.0001)) / 0.0001)
#     while prev_diff > next_diff:
#         prev_diff = next_diff
#         t = t - 0.01
#         next_diff = np.abs((f(x1 - t * gradient[0], x2 - t * gradient[1]) - f(x1 - t * gradient[0] + 0.0001, x2 - t * gradient[1] + 0.0001)) / 0.0001)
#     print('prev_diff: ', prev_diff)
#     print('next_diff: ', next_diff)
#     print('t: ', t)
#     print(f(x1 - t * gradient[0], x2 - t * gradient[1]))
#     print(f(x1, x2))
#     print('differece: ', f(x1 - t * gradient[0], x2 - t * gradient[1]) - f(x1, x2))
#     print()
    
#     x1 = x1 - t * gradient[0]
#     x2 = x2 - t * gradient[1]
#     if abs(gradient[0]) < 0.0001 and abs(gradient[1]) < 0.0001:
#         break

# print('x1: ', x1)
# print('x2: ', x2)