#!/bin/bash

# Run Python files in parallel
python3 test_grid.py & 
python3 test_grid1.py & 
python test_grid1.py & 

# Wait for all background processes to finish
wait

echo "All scripts have completed."
