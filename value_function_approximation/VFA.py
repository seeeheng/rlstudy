import numpy as np

class ValueFunctionApproximator:
    def __init__(self, n_states, n_actions, implementation):
        self.n_states = n_states
        self.n_actions = n_actions
        self.weights = np.zeros((n_states,n_actions))
        self.implementation = implementation
        self.implementation_tuple = ("MC", "TD(0)", "TD(lambda)")
        print("Oracle Implementation {} chosen.".format(implementation_tuple[implementation]))
    
    def oracle(self):
        # Returns the oraclized version of the value function
        if self.implementation == 0:
            pass
        # MC

        elif self.implementation == 1:
            target = !reward + !gamma*!qSt+1
            return target

        elif self.implementation == 2:
            pass

        else:
            raise

    def update_weights(self):
        # delw q(S,A,w) = x(s,A)
        # deltaW = alpha(qpi(S,A) - q(S,A,w))*x(S,A)