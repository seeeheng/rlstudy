import scipy.signal

def discount_cumsum(x, discount):
    """ Magic from rllab for computing discounted cumulative sums of vectors.
    
    Args: 
        vector x, 
        [x0, 
         x1, 
         x2]
    Returns:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def mlp(nn_shape, activation=nn.Tanh, output_activation=nn.Identity):
    """ Builds a feedforward neural network based on nn_shape.
    
    Uses nn.Sequential

    Args:
        nn_shape: [state dimensions, hiddenlayer1, ..., hiddenlayert, action dimensions]
        activation: activation function for hidden layers
        activation: activation function for output layer

    Returns:
        a MLP - nn.Sequential of a list of layers from input to output.
    """
    layers = []

    for i_current_layer in range(len(nn_shape)-1):
        current_activation = activation if i_current_layer < len(nn_shape)-2 else output_activation
        current_layer = nn_shape[i_current_layer]
        next_layer = nn_shape[i_current_layer + 1]
        layers += [nn.Linear(current_layer, next_layer), current_activation()]
    return nn.Sequential(*layers)