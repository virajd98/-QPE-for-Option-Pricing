# Piecewise Amplitude Loading
import classiq
from classiq.builtin_functions import PiecewiseLinearAmplitudeLoading
from classiq import Model, synthesize, show
from classiq import FunctionGenerator, FunctionLibrary, RegisterUserInput
from classiq import QUInt, synthesize
from classiq.builtin_functions import LinearPauliRotations, StatePreparation
import numpy as np

sp_num_qubits = 3

qmci_library = FunctionLibrary()
function_generator =  FunctionGenerator(function_name="state_loading")
# io = RegisterUserInput(size=3)
input_dict = function_generator.create_inputs(
    {"io": QUInt[sp_num_qubits], "ind": QUInt[1]}
)
#
# print(input_dict)

probabilities = np.linspace(0, 1, 2**sp_num_qubits) / sum(
    np.linspace(0, 1, 2**sp_num_qubits)
)

#can be specified, this is just an example
#for a better code, I will make the peobabilities list a user input and give options of usual distributions like Gaussian, log-normal etc

# print(probabilities)
sp_params = StatePreparation(
    probabilities=probabilities, error_metric={"KL": {"upper_bound": 0.00}}
)

sp_output = function_generator.StatePreparation(
    params=sp_params, strict_zero_ios=False, in_wires={"IN": input_dict["io"]})
print(sp_output)

function_generator.set_outputs({"io": sp_output["OUT"], "ind": input_dict["ind"]})
qmci_library.add_function(function_generator.to_function_definition())

function_generator = FunctionGenerator(function_name="amp_load")

# io = RegisterUserInput(size=3)
input_dict = function_generator.create_inputs(
    {"io": QUInt[sp_num_qubits], "ind": QUInt[1]}
)
amplitude_loading_params = PiecewiseLinearAmplitudeLoading(
    num_qubits=3,
    breakpoints=[0.5, 3.14, 5.0],
    affine_maps=[{"offset": 0, "slope": 0}, {"offset": -3.14, "slope": 1.6}],
    rescaling_factor=0.001,
)

#breakpoints=[s0 * x[0], k, s0 * x[-1]],
al_output = function_generator.PiecewiseLinearAmplitudeLoading(
    params=amplitude_loading_params, strict_zero_ios=False, in_wires={"state":input_dict["io"], "target": input_dict["ind"]}
)
print(al_output)
scaled_expectation_value = 0.7  # Probability of 1 after some execution
expectation_value = amplitude_loading_params.compute_expectation_value(
    scaled_expectation_value
)
function_generator.set_outputs({"io": al_output["state"], "ind": al_output["target"]})
qmci_library.add_function(function_generator.to_function_definition())

from classiq.builtin_functions import ZGate

function_generator = FunctionGenerator(function_name="good_state_oracle")

input_dict = function_generator.create_inputs(
    {"io": QUInt[sp_num_qubits], "ind": QUInt[1]}
)

z_out = function_generator.ZGate(
    params=ZGate(),
    in_wires={"TARGET": input_dict["ind"]},
)

function_generator.set_outputs({"ind": z_out["TARGET"], "io": input_dict["io"]})
qmci_library.add_function(function_generator.to_function_definition())

#Function representing reflection about the zero state

from classiq.builtin_functions import XGate
from classiq import ControlState
from classiq import QReg

function_generator = FunctionGenerator(function_name="zero_oracle")

reg_size = sp_num_qubits + 1
input_dict = function_generator.create_inputs({"mcz_io": QUInt[reg_size]})

x_out = function_generator.XGate(
    params=XGate(),
    in_wires={"TARGET": input_dict["mcz_io"][0]},
    should_control=False,
)

control_states = ControlState(ctrl_state="0" * (reg_size - 1), name="ctrl_reg")

mcz_out = function_generator.ZGate(
    params=ZGate(),
    control_states=control_states,
    in_wires={"TARGET": x_out["TARGET"], "ctrl_reg": input_dict["mcz_io"][1:reg_size]},
)

x_out = function_generator.XGate(
    params=XGate(), in_wires={"TARGET": mcz_out["TARGET"]}, should_control=False
)

function_generator.set_outputs(
    {"mcz_io": QReg.concat(x_out["TARGET"], mcz_out["ctrl_reg"])}
)

qmci_library.add_function(function_generator.to_function_definition())

from classiq.builtin_functions import UGate

## composite for Grover Diffuser

function_generator = FunctionGenerator(function_name="grover")
function_generator.include_library(qmci_library)


in_wires = function_generator.create_inputs(
    {"io": QUInt[sp_num_qubits], "ind": QUInt[1]}
)

oracle_out = function_generator.good_state_oracle(in_wires=in_wires)

pay_off_load_inv= function_generator.amp_load(in_wires={"io": oracle_out["io"], "ind": oracle_out["ind"]})

sps_inverse_out = function_generator.state_loading(
    in_wires={"io": pay_off_load_inv["io"], "ind": pay_off_load_inv["ind"]},
    is_inverse=True,
    should_control=False,
)

zero_oracle_out = function_generator.zero_oracle(
    in_wires={"mcz_io": QReg.concat(sps_inverse_out["io"], sps_inverse_out["ind"])}
)

sps_out = function_generator.state_loading(
    in_wires={
        "io": zero_oracle_out["mcz_io"][0:sp_num_qubits],
        "ind": zero_oracle_out["mcz_io"][sp_num_qubits],
    },
    should_control=False,
)

pay_off_load= function_generator.amp_load(in_wires={"io": sps_out["io"], "ind": sps_out["ind"]})

global_phase_out = function_generator.UGate(
    UGate(theta=0, phi=0, lam=0, gam=np.pi), in_wires={"TARGET": pay_off_load["ind"]}
)

function_generator.set_outputs({"io": pay_off_load["io"], "ind": global_phase_out["TARGET"]})

qmci_library.add_function(function_generator.to_function_definition())

#APPLYING AMPLITUDE ESTIMATION (AE) WITH QUANTUM PHASE ESTIMATION (QPE)
from classiq import Model
from classiq.builtin_functions import PhaseEstimation
from classiq.model import Constraints

n_qpe = 5
model = Model(constraints=Constraints(max_width=11))
model.include_library(qmci_library)
sp_output = model.state_loading()
al_output= model.amp_load(in_wires={"io": sp_output["io"], "ind": sp_output["ind"]})

qpe_out = model.PhaseEstimation(
    params=PhaseEstimation(
        size=n_qpe, unitary_params=qmci_library.get_function("grover"), unitary="grover"
    ),
    in_wires={"io": al_output["io"], "ind": al_output["ind"]},
)

model.set_outputs({"phase_result": qpe_out["PHASE_ESTIMATION"]})
qprog = synthesize(model.get_model())
# show(qprog)


import matplotlib.pyplot as plt

from classiq import execute

results = execute(qprog).result()

from classiq.execution import ExecutionDetails

res = results[0].value

phases_counts = res.parsed_counts

## mapping between register string to phases
phases_counts = dict(
    (sampled_state.state["phase_result"] / 2**n_qpe, sampled_state.shots)
    for sampled_state in res.parsed_counts
)
# print(phases_counts)

plt.bar(phases_counts.keys(), phases_counts.values(), width=0.1)
plt.xticks(rotation=90)
plt.show()
print("phase with max probability: ", max(phases_counts, key=phases_counts.get))

print(
    "measured amplitude/PAYOFF: ",
    np.sin(np.pi * max(phases_counts, key=phases_counts.get)) ** 2,
)

#Expected Outcome

sp_num_qubits= 3
probabilities = np.linspace(0, 1, 2**sp_num_qubits) / sum(
    np.linspace(0, 1, 2**sp_num_qubits)
)
k=3.14
s0= 20
expected_expectation_value = sum(
    [probabilities[x] * max(s0 * probabilities[x] - k, 0) for x in range(len(probabilities))]
)
print('EXPECTED PAYOFF:', expected_expectation_value)