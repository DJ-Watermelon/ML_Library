#!/bin/bash

# Loop through all test files and execute them
for test_file in "./tests/test*.json; do
    python3 ./run_test.py "{$test_file}"
done

echo "All tests completed."