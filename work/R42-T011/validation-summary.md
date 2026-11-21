# R42-T011 Validation Results

## Expected results

All six synthetic fixture records should pass validation.
Each invalid test case should produce a validation error.
Duplicate record IDs should be reported.

## Actual results

Command:
python3 -m unittest discover -s course-labs/r42-t011 -p "test_*.py" -v

Result: 6 tests passed, 0 failures, 0 errors.

| Test | Expected | Actual |
|---|---|---|
| Six valid fixture records | Pass | Pass |
| Missing record ID | Error reported | Pass |
| Invalid text | Error reported | Pass |
| Invalid numeric label | Error reported | Pass |
| Tensor-like label | Error reported | Pass |
| Duplicate record ID | Error reported | Pass |

All expected outcomes matched the actual test results.
