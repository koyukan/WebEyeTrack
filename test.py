from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

# Define the simulator function (loss function) to minimize
def simulator_of_nn(params):
    learning_rate_pow, num_neurons, type_neuron = params

    # Penalty based on the type of neuron
    neuron_penalty = {
        'relu': -1.0,
        'tanh': 0.0,
        'sin': 2.0
    }[type_neuron]

    # Compute the "loss"
    loss = (learning_rate_pow + 3)**2 + ((num_neurons - 64) / 64)**2 + neuron_penalty
    return loss

# Define the search space
dimensions = [
    Real(-5.0, -1.0, name='learning_rate_pow'),
    Integer(2, 256, name='num_neurons'),
    Categorical(['relu', 'tanh', 'sin'], name='type_neuron')
]

# Perform the optimization
result = gp_minimize(
    func=simulator_of_nn,  # Objective function
    dimensions=dimensions,  # Search space
    n_calls=11,  # Number of calls to evaluate the function
    random_state=42  # Random seed for reproducibility
)

# Output the results
print("Optimized Parameters:", result.x)
print("Minimum Loss:", result.fun)
