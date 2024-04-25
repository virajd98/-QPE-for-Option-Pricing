from classiq import Model, synthesize, show, set_constraints, execute
from classiq.builtin_functions import ArithmeticOracle, GroverOperator, StatePreparation
from classiq.execution import QaeWithQpeEstimationMethod
from classiq.model import Constraints, OptimizationParameter
oracle_params = ArithmeticOracle(
    expression="b+ c == a",
    definitions=dict(b=dict(size=1),c=dict(size=1),a=dict(size=2)),
)
state_preparation_params = StatePreparation(
    probabilities=[0.25,0,0,0,0,0.25,0,0,0,0,0.05,0.20,0,0,0.25,0],
    error_metric={"KL": {"upper_bound": 0}},
)
grover_operator_params = GroverOperator(
    oracle_params=oracle_params,
    state_preparation_params=state_preparation_params,
)
model = Model()
model.StatePreparation(params=state_preparation_params)
GROVER_OUTPUT= model.GroverOperator(params=grover_operator_params)
model.set_outputs({"a": GROVER_OUTPUT["a"]})
model.set_outputs({"b": GROVER_OUTPUT["b"]})
model.set_outputs({"c": GROVER_OUTPUT["c"]})
model.sample()
constraints = Constraints(max_width=25)
serialized_model = model.get_model()
serialized_model = set_constraints(serialized_model, constraints)
quantum_program = synthesize(serialized_model)
# show(quantum_program)

import numpy as np

def get_counts(quantum_program):
    job = execute(quantum_program) #job is like an id, can be used later
    results = job.result()
    return results[0].value.counts, results[0].value.parsed_counts


_, parsed_counts = get_counts(quantum_program)
print(parsed_counts)

