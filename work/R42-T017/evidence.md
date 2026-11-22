# R42-T017 Evidence

## Environment
- Python: 3.14.6
- Dependencies: Python standard library only
- Execution: Local CPU-only

## Inputs
- course-labs/r42-t017/config.json
- course-labs/r42-t017/fixture.json

## Configuration
- Schema version: 1.0
- Seed: 42
- Feature order: row-major flattened pixels
- Algorithm: nonzero_pixel_baseline
- Maximum samples: 6
- Maximum steps: 1

## Automated tests
Command:
python3 -m unittest discover -s course-labs/r42-t017 -p "test_*.py"

Result: 11 tests passed.

## Pipeline results

| Split | Expected | Actual |
|---|---|---|
| Train | [0, 0] | [0, 0] |
| Validation | [0, 1] | [0, 1] |
| Test | [1, 1] | [1, 1] |

All predictions matched.

## Limitations
Synthetic fixture verification only. No real satellite imagery, trained CNN, or real-world ship-detection performance is claimed.
