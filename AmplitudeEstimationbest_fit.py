import classiq
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
# classiq.authenticate()
from classiq import Model, synthesize, show, set_constraints, execute
from classiq.builtin_functions import (
    AmplitudeEstimation,
    ArithmeticOracle,
    GroverOperator,
    StatePreparation,
)
from classiq.execution import QaeWithQpeEstimationMethod
from classiq.model import Constraints, OptimizationParameter

oracle_params = ArithmeticOracle(
    expression="b + c == a",
    definitions=dict(b=dict(size=1),c=dict(size=1), a=dict(size=2)),
)

#0+0=00 (state: 0000 = 0)
#0+1=01 (state: 0110= 6)
#1+0= 01 (state: 0101= 5)
#1+1=10 (state: 1011= 11)

state_preparation_params = StatePreparation(
    probabilities=[0.25,0,0,0,0,0,0,0.05,0.2,0,0,0.5,0,0,0,0],
    error_metric={"KL": {"upper_bound": 0}},
)   
grover_operator_params = GroverOperator(
    oracle_params=oracle_params,
    state_preparation_params=state_preparation_params,
)
qae_params = AmplitudeEstimation(
    grover_operator=grover_operator_params,
    estimation_register_size=4,
)
model = Model()
qae_out_wires = model.AmplitudeEstimation(params=qae_params)
model.set_outputs({"phase": qae_out_wires["ESTIMATED_AMPLITUDE_OUTPUT"]})
model.sample()
model.post_process_amplitude_estimation(
    estimation_register_size=4, estimation_method=QaeWithQpeEstimationMethod.BEST_FIT)
constraints = Constraints(max_width=25)
# constraints = Constraints(optimization_parameter=OptimizationParameter.WIDTH)
serialized_model = model.get_model()
serialized_model = set_constraints(serialized_model, constraints)
# print(serialized_model )
quantum_program = synthesize(serialized_model)
# show(quantum_program)

# qae_result = execute(quantum_program).result()
# print(qae_result)

# print(f"The probability estimation of the good states is (best_fit method): {qae_result[1].value}")

# print(f"The accurate expected result is: {0.75}")


# counts = sorted(qae_result[0].value.counts_by_qubit_order(lsb_right=True).items())
# print(counts)
# num_shots = sum(count[1] for count in counts)
# print(num_shots)
# print(
#     f"probabilities are:\n{dict([(bit_string, count/num_shots) for bit_string, count in counts])}"
# )

# #try
# # QaeWithQpeEstimationMethod.MAXIMUM_LIKELIHOOD

def get_counts(quantum_program):
    job = execute(quantum_program) #job is like an id, can be used later
    results = job.result()
    return results[0].value.counts, results[0].value.parsed_counts

def process_parsed_counts(parsed_counts):
    y = []
    shots = []
    for item in parsed_counts:
        y.append(item.state['phase'])
        shots.append(item.shots)
    return np.array(y), np.array(shots)

    
def generate_histogram(y, prob, figsize=(10, 5)):
    fig = plt.figure(figsize = figsize)

    plt.scatter(y, prob)
    return plt.show()

def multiple_fitting_curve_parameters(recording_qubits, data_y, data_prob, guesses, bounds = (0, 1)):
    
    M = 2 ** recording_qubits
    
    def prob_func(y, a0, a1, theta0, theta1):
        coeff = 1 / (M**2)
        
        angle0 = y - (theta0 * M)
        numerator0 = 1 - np.cos(2 * math.pi * angle0)
        denominator0 = 1 - np.cos(2 * math.pi * angle0 / M)

        angle1 = y - (theta1 * M)
        numerator1 = 1 - np.cos(2 * math.pi * angle1)
        denominator1 = 1 - np.cos(2 * math.pi * angle1 / M)
        
        prob_y = coeff * (a0 * numerator0/denominator0 + a1 * numerator1/denominator1 )
        return prob_y
    
    guess1 = [0.5, 0.5, (guesses[0] - 0.5)/M, (guesses[1] - 0.5)/M]
    guess2 = [0.5, 0.5, (guesses[0] + 0.5)/M, (guesses[1] + 0.5)/M]
    guess3 = [0.5, 0.5, (guesses[0] - 0.5)/M, (guesses[1] + 0.5)/M]
    guess4 = [0.5, 0.5, (guesses[0] + 0.5)/M, (guesses[1] - 0.5)/M]

    bounds = ([0, 0,(guesses[0] - 0.5)/M , (guesses[1] - 0.5)/M], [1, 1, (guesses[0] + 0.5)/M, (guesses[1] + 0.5)/M])
    
    try: 
        parameters1, pcov1 = curve_fit(prob_func, data_y, data_prob, bounds = bounds, p0 = guess1) # max_nfev = 1000
    except ValueError:
        pcov1 = np.inf
    
    try:
        parameters2, pcov2 = curve_fit(prob_func, data_y, data_prob, bounds = bounds, p0 = guess2)
    except ValueError:
        pcov2 = np.inf
    
    try:
        parameters3, pcov3 = curve_fit(prob_func, data_y, data_prob, bounds = bounds, p0 = guess3)
    except ValueError:
        pcov3 = np.inf
    
    try:
        parameters4, pcov4 = curve_fit(prob_func, data_y, data_prob, bounds = bounds, p0 = guess4)
    except ValueError:
        pcov4 = np.inf


    parameters_list = [parameters1, parameters2, parameters3, parameters4]
    pcov_sum_list = [np.sum(pcov1), np.sum(pcov2), np.sum(pcov3), np.sum(pcov4)]
    pcov_list = [pcov1, pcov2, pcov3, pcov4]

    if np.isinf(pcov_sum_list[0]) and np.isinf(pcov_sum_list[1]) and np.isinf(pcov_sum_list[2]) and np.isinf(pcov_sum_list[3]):
        print("No optimal parameters found")
        return False, False
    else:
        index = pcov_sum_list.index(min(pcov_sum_list))
        parameters = parameters_list[index]
        pcov = pcov_list[index]

    range_y = np.linspace(0, M, 1000)
    probabilities = np.array(prob_func(range_y, parameters[0], parameters[1], parameters[2], parameters[3]))
    #probabilities = np.nan_to_num(probabilities, nan=1 / (M**2))
    plt.plot(range_y, probabilities)

    # max_value_index = np.argmax(probabilities)
    # result = range_y[max_value_index]

    plt.show()
    return parameters, pcov

import math
recording_qubits= 4
_, parsed_counts = get_counts(quantum_program)
print('pc:', parsed_counts)
# print(type(parsed_counts))
data_y, shots = process_parsed_counts(parsed_counts)
print('ds:', data_y,shots)
guesses = data_y[:2]
print('guesses', guesses)
data_prob = (1/2048) * np.array(shots)
generate_histogram(data_y,data_prob)

parameters, pcov= multiple_fitting_curve_parameters(recording_qubits, data_y, data_prob, guesses, bounds = (0, 1))
print(parameters)

z=[]
k=[]
x= np.sin(np.pi *parameters[2]) ** 2
z.append(x)
y= np.sin(np.pi *parameters[3]) ** 2
z.append(y)

k.append(parameters[0])
k.append(parameters[1])

print(z)
print(k)






