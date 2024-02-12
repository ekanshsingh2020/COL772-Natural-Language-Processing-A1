#!bin/bash

# Run the model

flag="$1"

if [ $flag == "train" ]; then
    python3 final_trained.py $2 $3
else
    python3 final_testing.py $2 $3 $4
fi