use anyhow::{Result, Context};
use wasmtime::component::{Component, Linker, InstancePre};
use wasmtime::{Config, Engine, Store};

wasmtime::component::bindgen!({
    // For running the client.
    world: "client",
    path: "../matrix-client/wit/world.wit", // Path to client's world WIT
    additional_packages: [ // Dependencies of client's world
        { package = "wasi-custom:host-offload@0.1.0", path = "../wit/host-offload.wit" },
    ],
    interface_imports: true, 
});

// We don't strictly need to generate bindings for the provider's world if we are just
// taking its instance and passing it to the linker. However, if we wanted to
// call its functions from the host, we would.
// For linking, Wasmtime primarily needs the component instance that exports the interface.


fn main() -> Result<()> {
    println!("[Runner] Setting up Wasmtime engine and store...");
    let mut config = Config::new();
    config.wasm_component_model(true);
    
    let engine = Engine::new(&config)?;
    let mut store = Store::new(&engine, ()); // No complex host state needed for this runner

    // --- Load Provider Component ---
    let provider_component_path = "../host-offload-provider/target/wasm32-unknown-unknown/release/host_offload_provider.wasm";
    println!("[Runner] Loading provider component from: {}", provider_component_path);
    let provider_component = Component::from_file(&engine, provider_component_path)
        .context("Failed to load provider component")?;

    // --- Load Client Component ---
    let client_component_path = "../matrix-client/target/wasm32-unknown-unknown/release/matrix_client.wasm";
    println!("[Runner] Loading client component from: {}", client_component_path);
    let client_component = Component::from_file(&engine, client_component_path)
        .context("Failed to load client component")?;


    // --- Link Components ---
    // The client component imports "host-allocator".
    // The provider component exports "host-allocator".
    // We need to tell the linker for the client how to satisfy this import.

    let mut linker = Linker::new(&engine);

    let provider_instance_pre: InstancePre<()> = linker.instantiate_pre(&provider_component)
        .context("Failed to pre-instantiate provider component")?;
    
    client::add_to_linker_imports(&mut linker, |_, name: &str| {
         match name {
            "host-allocator" => Ok(provider_instance_pre.clone()), // Provide the pre-instantiated provider?
            _ => anyhow::bail!("Unknown import: {}", name),
        }
    })?;

    println!("[Runner] Instantiating client component and linking with provider...");
    // Instantiate the client, which will in turn fully instantiate the provider
    // because `provider_instance_pre` is now linked.
    let (client_instance, _) = client::Client::instantiate_pre(&mut store, &client_component, &linker)
         .context("Failed to instantiate client component with provider")?;


    // --- Calling the Client's Exported Function ---
    println!("[Runner] Calling 'run-matrix-example' in client Wasm...");
    match client_instance.call_run_matrix_example(&mut store) {
        Ok(Ok(_)) => println!("[Runner] 'run-matrix-example' executed successfully."),
        Ok(Err(e)) => eprintln!("[Runner] 'run-matrix-example' in client returned an error: {}", e),
        Err(e) => eprintln!("[Runner] Trap during 'run-matrix-example' in client: {}", e),
    }

    Ok(())
}
