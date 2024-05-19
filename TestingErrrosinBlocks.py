import numpy as np
import random


# probabilities= [0.04,0.20,0.17,0.25,0.08,0.05,0.18,0.03] 
probabilities= [0.20, 0.35, 0.25, 0.20] #probabilities of each price paths

k=70
s0= 50
Y= [0.9, 1.18, 1.47, 1.75]  # s0*Y[i] represents the possible price path for i in range(4)
 
# Y= [0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3]  

# s0*Y[i] represents the possible price path for i in range(4)
pay_off_dist = [probabilities[x] * max ((s0 * Y[x]) - k , 0) for x in range(len(probabilities))]
pay_offs = [max ((s0 * Y[x]) - k , 0) for x in range(len(probabilities))]
print(pay_offs)

#E_P[f(X)]
expected_expectation_value= np.sum(pay_off_dist)

print('EXPECTED PAYOFF:', expected_expectation_value)



#Finance Model

#importing necessary modules 

import classiq
from classiq.builtin_functions import PiecewiseLinearAmplitudeLoading, PhaseEstimation, PiecewiseLinearRotationAmplitudeLoading
from classiq import QUInt, Model, synthesize, show, QReg, ControlState, execute,set_constraints, set_execution_preferences, OptimizationParameter
from classiq import FunctionGenerator, FunctionLibrary, RegisterUserInput
from classiq.builtin_functions import LinearPauliRotations, StatePreparation
from classiq.builtin_functions import XGate, UGate, ZGate
from classiq.model import Constraints
from classiq.execution import ExecutionDetails, ExecutionPreferences, ClassiqBackendPreferences
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from classiq import set_preferences
from classiq.model import Preferences

qmci_library = FunctionLibrary()
function_generator =  FunctionGenerator(function_name="state_amp_load")

sp_num_qubits = 2

#number of possible option prices is equal to 2^{number of qubits}. Here we consider 4 possible option price paths

input_dict = function_generator.create_inputs({"io": QUInt[sp_num_qubits], "ind": QUInt[1]})



sp_params = StatePreparation(
    probabilities=probabilities, error_metric={"KL": {"upper_bound": 0.00}}
)

sp_output = function_generator.StatePreparation(
    params=sp_params, strict_zero_ios=False, in_wires={"IN": input_dict["io"]})
#the in_wires argument soecifies the input register to the StatePreparation operation

amplitude_loading_params = PiecewiseLinearAmplitudeLoading(
    num_qubits=2,
    breakpoints=[45, 70, 87.5],
    affine_maps=[{"offset": 0, "slope": 0}, {"offset": -70, "slope": 1}],
    rescaling_factor=0.05)

al_output = function_generator.PiecewiseLinearAmplitudeLoading(
    params=amplitude_loading_params, strict_zero_ios=False, in_wires={"state":sp_output["OUT"], "target": input_dict["ind"]}
)
function_generator.set_outputs({"io": al_output["state"], "ind": al_output["target"]})

qmci_library.add_function(function_generator.to_function_definition())


#APPLYING AMPLITUDE ESTIMATION (AE) WITH QUANTUM PHASE ESTIMATION (QPE)

n_qpe = 3
constraints = Constraints(
    max_width=30)
# optimization_parameter=OptimizationParameter.DEPTH
# preferences = Preferences(timeout_seconds= 4000, optimization_timeout_seconds= 2000)
preferences = Preferences(timeout_seconds= 7000)

# backend_preferences = ClassiqBackendPreferences(backend_name="nvidia_state_vector_simulator")
# execution_preferences=ExecutionPreferences(num_shots=10000, backend_preferences=backend_preferences)


execution_preferences=ExecutionPreferences(num_shots=100000)
model = Model()
model.include_library(qmci_library)
sp_output = model.state_amp_load() 
print(sp_output)
# qpe_out = model.PhaseEstimation(
#     params=PhaseEstimation(
#         size=n_qpe, unitary_params=qmci_library.get_function("grover"), unitary="grover"
#     ),
#     in_wires={"io": sp_output["io"], "ind": sp_output["ind"]},
# )

model.set_outputs({"state": sp_output["io"], "target": sp_output["ind"]})
serialized_model = model.get_model()
serialized_model_constraints = set_constraints(serialized_model,constraints)
serialized_model_pref = set_preferences(serialized_model_constraints, preferences)
serialized_model_ex_pref = set_execution_preferences(serialized_model_pref, execution_preferences)
qprog= synthesize(serialized_model_ex_pref)
show(qprog)


