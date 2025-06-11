#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Building Provider Component (after fixing matrix bug) ---"
(cd host-offload-provider && cargo component build --target wasm32-wasip2)

echo
echo "--- Building Client Component ---"
(cd matrix-client && cargo component build --target wasm32-wasip2)

echo
echo "--- Composing Components with 'wac' ---"
wac plug \
  matrix-client/target/wasm32-wasip2/debug/matrix_client.wasm \
  --plug host-offload-provider/target/wasm32-wasip2/debug/host_offload_provider.wasm \
  -o composed.wasm

echo
echo "--- Running Composed Component in Wasmtime ---"


wasmtime run --invoke 'run-matrix-example()' composed.wasm
echo
echo "--- Script finished successfully! ---"
