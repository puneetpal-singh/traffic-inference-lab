from traffic_lab.inspection import weighted_input_types


def test_weighted_precision_evidence_uses_executed_tensors_and_ignores_registration():
    events = [
        {
            "args": {
                "provider": "CPUExecutionProvider",
                "op_name": "Conv",
                "input_type_shape": [{"float": [1, 3, 32, 32]}, {"float": [8, 3, 3, 3]}],
            }
        },
        {
            "args": {
                "provider": "CUDAExecutionProvider",
                "op_name": "MatMul",
                "input_type_shape": [{"float16": [8, 8]}],
            }
        },
        {
            "args": {
                "provider": "CPUExecutionProvider",
                "op_name": "Cast",
                "input_type_shape": [{"float16": [8]}],
            }
        },
        {"args": {"op_name": "Conv", "input_type_shape": [{"float16": [8]}]}},
    ]
    assert weighted_input_types(events) == {
        "CPUExecutionProvider:float": 1,
        "CUDAExecutionProvider:float16": 1,
    }
