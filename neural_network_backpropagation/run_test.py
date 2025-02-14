import sys
import json
import operations
import neural_net
import numpy as np

def test_activation(test_case, test_details):
    activation = test_details["activation"]
    run_value = test_details['value'] == 'true'
    run_derivative = test_details['value'] == 'false'

    assert(run_value or run_derivative)

    if activation == "relu":
        x = np.array(test_details["x"])
        soln = np.array(test_details["soln"])

        if run_value:
            res = operations.ReLU().value(x)
        else:
            res = operations.ReLU().derivative(x)
        
        if np.array_equiv(res, soln):
            print(f'Test case {test_case} passed')
        else:
            print(f'Test case {test_case} failed')
            print(f'Expected: {soln}')
            print(f'Got: {res}')
        
    elif activation == "sigmoid":
        k = test_details["k"]
        x = np.array(test_details["x"])
        soln = np.array(test_details["soln"])

        if run_value:
            res = operations.Sigmoid(k).value(x)
        else:
            res = operations.Sigmoid(k).derivative(x)

        if np.array_equiv(res, soln):
            print(f'Test case {test_case} passed')
        else:
            print(f'Test case {test_case} failed')
            print(f'Expected: {soln}')
            print(f'Got: {res}')
    else:
        raise Exception(f'Invalid activation function for test {test_case}')

def test_metric(test_case, test_details):
    metric = test_details["metric"]
    y_hat = np.array(test_details["y_hat"])
    y = np.array(test_details["y"])
    soln = test_details["soln"]

    if metric == "mean_absolute_error":
        res = operations.mean_absolute_error(y_hat, y)

        if res == soln:
            print(f'Test case {test_case} passed')
        else:
            print(f'Test case {test_case} failed')
            print(f'Expected: {soln}')
            print(f'Got: {res}')
    else:
        raise Exception(f'Invalid metric for test {test_case}')
    
def get_activation(name):
    if name == 'identity':
        return operations.Identity()
    elif name == 'sigmoid':
        return operations.Sigmoid()
    elif name == 'relu':
        return operations.ReLU()
    else:
        raise Exception(f'Invalid activation function')

def test_forward(test_case, test_details, networks):
    network_name = test_details['net']
    network_details = networks[network_name]
    n_features = network_details['n_features']
    layer_sizes = network_details['layer_sizes']
    activations = network_details['activations']
    activations = [get_activation(a) for a in activations]
    loss = network_details['loss']
    learning_rate = network_details['learning_rate']
    W_init = network_details['W_init']
    W_init = [np.array(w) for w in W_init]

    X = np.array(test_details['X'])
    soln_A_vals, soln_Z_vals = np.array(test_details['soln'])
    soln_A_vals = [np.array(a) for a in soln_A_vals]
    soln_Z_vals = [np.array(z) for z in soln_Z_vals]

    nn = neural_net.NeuralNetwork(n_features=n_features, layer_sizes=layer_sizes, activations=activations, loss=loss, learning_rate=learning_rate, W_init=W_init)

    A_vals, Z_vals = nn.forward_pass(X)

    if A_vals and Z_vals and np.allclose(A_vals, soln_A_vals, atol=1e-10) and np.allclose(Z_vals, soln_Z_vals, atol=1e-10):
        print(f'Test case {test_case} passed')
    else:
        print(f'Test case {test_case} failed')
        if not np.allclose(A_vals, soln_A_vals, atol=1e-10):
            print(f'Expected A_vals: {soln_A_vals}')
            print(f'Got A_vals: {A_vals}')
        if not np.allclose(Z_vals, soln_Z_vals, atol=1e-10):
            print(f'Expected Z_vals: {soln_Z_vals}')
            print(f'Got Z_vals: {Z_vals}')

def get_networks(network_init_file_path):
    with open(network_init_file_path, 'r') as network_init_file:
        text = network_init_file.read()
        network_json = json.loads(text)

        return network_json
    
def test_backward(test_case, test_details, networks):
    network_name = test_details['net']
    network_details = networks[network_name]
    n_features = network_details['n_features']
    layer_sizes = network_details['layer_sizes']
    activations = network_details['activations']
    activations = [get_activation(a) for a in activations]
    loss = network_details['loss']
    learning_rate = network_details['learning_rate']
    W_init = network_details['W_init']
    W_init = [np.array(w) for w in W_init]

    A_vals = test_details['A_vals']
    A_vals = [np.array(a) for a in A_vals]
    dLdyhat = np.array(test_details['dLdyhat'])

    soln = test_details['soln']

    nn = neural_net.NeuralNetwork(n_features=n_features, layer_sizes=layer_sizes, activations=activations, loss=loss, learning_rate=learning_rate, W_init=W_init)

    res = nn.backward_pass(A_vals, dLdyhat)

    if np.allclose(res, soln, atol=1e-10):
        print(f'Test case {test_case} passed')
    else:
        print(f'Test case {test_case} failed')
        print(f'Expected: {soln}')
        print(f'Got: {res}')

def test_update_weights(test_case, test_details, networks):
    network_name = test_details['net']
    network_details = networks[network_name]
    n_features = network_details['n_features']
    layer_sizes = network_details['layer_sizes']
    activations = network_details['activations']
    activations = [get_activation(a) for a in activations]
    loss = network_details['loss']
    learning_rate = network_details['learning_rate']
    W_init = network_details['W_init']
    W_init = [np.array(w) for w in W_init]

    X = np.array(test_details['X'])
    Z_vals = test_details['Z_vals']
    Z_vals = [np.array(z) for z in Z_vals]
    deltas = test_details['deltas']
    deltas = [np.array(d) for d in deltas]

    soln = test_details['soln']

    nn = neural_net.NeuralNetwork(n_features=n_features, layer_sizes=layer_sizes, activations=activations, loss=loss, learning_rate=learning_rate, W_init=W_init)

    W = nn.update_weights(X, Z_vals, deltas)

    if np.allclose(W, soln, atol=1e-10):
        print(f'Test case {test_case} passed')
    else:
        print(f'Test case {test_case} failed')
        print(f'Expected: {soln}')
        print(f'Got: {W}')

def test_loss(test_case, test_details):
    loss = test_details["loss"]
    run_value = test_details['value'] == 'true'
    run_derivative = test_details['value'] == 'false'
    y_hat = np.array(test_details["y_hat"])
    y = np.array(test_details["y"])
    soln = test_details["soln"]

    assert(run_value or run_derivative)

    if loss == "mean_squared_error":
        if run_value:
            res = operations.MeanSquaredError().value(y_hat, y)
        else:
            res = operations.MeanSquaredError().derivative(y_hat, y)

        if np.allclose(res, soln, atol=1e-10):
            print(f'Test case {test_case} passed')
        else:
            print(f'Test case {test_case} failed')
            print(f'Expected: {soln}')
            print(f'Got: {res}')
    else:
        raise Exception(f'Invalid loss for test {test_case}')

def test(test_file_path, networks):
    with open(test_file_path, 'r') as test_file:
        text = test_file.read()
        test_json = json.loads(text)

        
        for test_case in test_json:
            print(f'Running test case: {test_case}')

            test_details = test_json[test_case]

            if "activation" in test_details:
                test_activation(test_case, test_details)
            elif "metric" in test_details:
                test_metric(test_case, test_details)
            elif "loss" in test_details:
                test_loss(test_case, test_details)
            elif "X" in test_details and "Z_vals" not in test_details:
                test_forward(test_case, test_details, networks)
            elif "A_vals" in test_details:
                test_backward(test_case, test_details, networks)
            elif "X" in test_details and "Z_vals" in test_details:
                test_update_weights(test_case, test_details, networks)
            else:
                raise Exception(f'Invalid test type for test {test_case}')
            print()


if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python run_test.py test_file_path [network_init_file_path]")
        sys.exit(1)

    test_file_path = sys.argv[1]
    network_init_file_path = './tests/nets.json'
    if len(sys.argv) > 2:
        network_init_file_path = sys.argv[2]

    networks = get_networks(network_init_file_path)
    
    test(test_file_path, networks)
