from VFA import ValueFunctionApproximator

class ValueFunctionApproximatorMC(ValueFunctionApproximator):
    def __init__(self):
        ValueFunctionApproximator.__init__(self, n_states, n_actions, 0)
    
    def oracle(self):
        pass 

    def update_weights(self, replay):
        # REPLAY will be of form
        # (state, action, reward, next_state)
        for state, action, reward, next_state in replay: