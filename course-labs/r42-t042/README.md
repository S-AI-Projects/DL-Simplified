# Course Lab R42-T042: Sign-Language MNIST - Deterministic Split Manifest

## Overview
This component implements a deterministic split manifest generator and split-leakage validator. It partitions the dataset into train, validation, and test splits with strict disjoint record ID guarantees.

## Synthetic Fixture Notice
The data in `fixture_data.json` contains six fictional text records (IDs t1..t6 with labels 0 and 1).
**Note:** This is a CPU smoke fixture designed strictly for pipeline verification and split isolation testing, NOT a real corpus or evidence of real-world performance.

## Running Tests
Run the unit tests:
```bash
python3 -m unittest discover -s course-labs/r42-t042 -p "test_*.py"
```
