#!/bin/bash

# Script to run federated learning experiments with different partitioning strategies
# Usage: ./run_federated.sh [iid|pathological_non_iid|label_skew]

STRATEGY=${1:-iid}
NUM_ROUNDS=10
NUM_CLIENTS=7
MIN_CLIENTS=7
LOCAL_EPOCHS=2
BATCH_SIZE=32
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
SERVER_ADDRESS="localhost:$PORT"

echo "=========================================="
echo "Running Federated Learning Experiment"
echo "=========================================="
echo "Strategy: $STRATEGY"
echo "Rounds: $NUM_ROUNDS"
echo "Clients: $NUM_CLIENTS"
echo "Local Epochs per Round: $LOCAL_EPOCHS"
echo "=========================================="
echo ""

# Check if partitions exist
if [ ! -d "partitions/$STRATEGY" ]; then
    echo "Error: Partitions for strategy '$STRATEGY' not found!"
    echo "Please run partition.py first to create the partitions."
    exit 1
fi

# Start the server in background
echo "Starting Flower server..."
python fl_server.py \
    --num_rounds $NUM_ROUNDS \
    --num_clients $NUM_CLIENTS \
    --min_clients $MIN_CLIENTS \
    --server_address $SERVER_ADDRESS \
    --partition_strategy $STRATEGY \
    --save_path "best_federated_${STRATEGY}.pth" &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
sleep 5  # Wait for server to initialize

# Start clients
echo ""
echo "Starting $NUM_CLIENTS clients..."
CLIENT_PIDS=()

for i in $(seq 0 $((NUM_CLIENTS-1))); do
    echo "Starting client $i..."
    python fl_client.py \
        --client_id $i \
        --partition_strategy $STRATEGY \
        --server_address $SERVER_ADDRESS \
        --batch_size $BATCH_SIZE \
        --local_epochs $LOCAL_EPOCHS &
    
    CLIENT_PIDS+=($!)
    sleep 1
done

echo ""
echo "All clients started. Waiting for training to complete..."
echo "Client PIDs: ${CLIENT_PIDS[@]}"

# Wait for all clients to finish
for pid in "${CLIENT_PIDS[@]}"; do
    wait $pid
done

# Wait for server to finish
wait $SERVER_PID

echo ""
echo "=========================================="
echo "Federated Learning Experiment Complete!"
echo "Strategy: $STRATEGY"
echo "=========================================="