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
    expression="a == b",
    definitions=dict(a=dict(size=2), b=dict(size=2)),
)
state_preparation_params = StatePreparation(
    probabilities=[0.25,0,0,0,0,0.25,0,0,0,0,0.25,0,0,0,0.25,0],
    error_metric={"KL": {"upper_bound": 0}},
)
grover_operator_params = GroverOperator(
    oracle_params=oracle_params,
    state_preparation_params=state_preparation_params,
)
qae_params = AmplitudeEstimation(
    grover_operator=grover_operator_params,
    estimation_register_size=2,
)
model = Model()
qae_out_wires = model.AmplitudeEstimation(params=qae_params)
model.set_outputs({"phase": qae_out_wires["ESTIMATED_AMPLITUDE_OUTPUT"]})
model.sample()
model.post_process_amplitude_estimation(
    estimation_register_size=2, estimation_method=QaeWithQpeEstimationMethod.MAXIMUM_LIKELIHOOD
)

# constraints = Constraints(optimization_parameter=OptimizationParameter.WIDTH)
serialized_model = model.get_model()
# serialized_model = set_constraints(serialized_model, constraints)
# print(serialized_model )
quantum_program = synthesize(serialized_model)
show(quantum_program)

qae_result = execute(quantum_program).result()
# print(qae_result)

print(f"The probability estimation of the good states is (Max_Likelihood Method"
      f"): {qae_result[1].value}")

print(f"The accurate expected result is: {0.75}")


counts = sorted(qae_result[0].value.counts_by_qubit_order(lsb_right=True).items())
print(counts)
num_shots = sum(count[1] for count in counts)
print(num_shots)
print(
    f"probabilities are:\n{dict([(bit_string, count/num_shots) for bit_string, count in counts])}"
)