# #!/bin/bash

# # Array of lambda values
# lambda_values=(0.0 1.0)

# # Base command
# base_command="python run.py --data_folder trajectory_dataset --setting train"

# # Loop through lambda values and run each in background
# for lambda in "${lambda_values[@]}"; do
#     echo "Starting training with lambda_value=$lambda"
#     $base_command --lambda_value "$lambda" > "logs/lambda_$lambda.log" 2>&1 &
# done

# # Wait for all background processes to finish
# wait
# echo "All training processes completed."
#!/bin/bash

# Array of lambda values
lambda_values=(0.0 1.0)

# Base command
base_command="python run.py --data_folder trajectory_dataset --setting train"

# Create logs directory if it doesn't exist
mkdir -p logs

# Loop through lambda values and run each in a new GNOME terminal tab
for lambda in "${lambda_values[@]}"; do
    echo "Opening new tab for lambda_value=$lambda"
    gnome-terminal --tab --title="lambda=$lambda" -- bash -c \
    "$base_command --lambda_value $lambda | tee logs/lambda_$lambda.log; exec bash"
done
