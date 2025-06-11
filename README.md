# wit-interface

This project demonstrates the WebAssembly (Wasm) Component Model by creating two components that communicate with each other:
1.  `matrix-client`: A component that wants to perform a matrix multiplication.
2.  `host-offload-provider`: A component that provides the matrix multiplication capability.

The `client` offloads the computation to the `provider` by allocating memory within the provider's context, copying data to it, and then calling a function using handles to that data, similar to a CUDA programming model.

## Prerequisites

1.  **Rust Toolchain**: Install Rust from [rust-lang.org](https://www.rust-lang.org/).
2.  **Wasm Target**: Add the `wasm32-wasip2` target. This is the modern target for the component model.
    ```sh
    rustup target add wasm32-wasip2
    ```
3.  **cargo-component**: Install the cargo subcommand for building components.
    ```sh
    cargo install cargo-component
    ```
4.  **wac-cli**: Install the Wasm Composition tool to link components together.
    ```sh
    cargo install wac-cli
    ```
5.  **Wasmtime**: Ensure you have a recent version of the `wasmtime` CLI installed.
    ```sh
    cargo install wasmtime-cli
    ```

## How to Build and Run

1.  **Build the Provider Component**:
    Navigate to the provider's directory and build it.
    ```sh
    cd host-offload-provider
    cargo component build --target wasm32-wasip2
    cd ..
    ```

2.  **Build the Client Component**:
    Navigate to the client's directory and build it.
    ```sh
    cd matrix-client
    cargo component build --target wasm32-wasip2
    cd ..
    ```

3.  **Compose the Components**:
    Use `wac` to "plug" the provider's implementation into the client's import. This creates a single, runnable Wasm component.
    ```sh
    wac plug matrix-client/target/wasm32-wasip2/debug/matrix_client.wasm --plug host-offload-provider/target/wasm32-wasip2/debug/host_offload_provider.wasm -o composed.wasm
    ```
    *Note: The first component listed is the "main" one that has an import to be satisfied. The `--plug` argument provides the implementation.*

4.  **Run the Composed Component**:
    Use `wasmtime` to run the `run-matrix-example` function exported from our composed component.
    ```sh
    wasmtime run --invoke 'run-matrix-example()' composed.wasm
    ```
