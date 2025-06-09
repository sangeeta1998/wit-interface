use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy; // For thread-safe static initialization

#[allow(warnings)]
mod bindings;

use bindings::exports::wasi_custom::host_offload::host_allocator::Guest;

use bindings::exports::wasi_custom::host_offload::host_allocator::{Handle, HostError, MatrixDimensions};

// Simulated host state
struct HostState {
    buffers: HashMap<Handle, Vec<u8>>,
    matrix_dims: HashMap<Handle, (u32, u32)>, // rows, cols
    next_handle: Handle,
}

impl HostState {
    fn new() -> Self {
        HostState {
            buffers: HashMap::new(),
            matrix_dims: HashMap::new(),
            next_handle: 1, // Start handles from 1
        }
    }

    fn new_handle(&mut self) -> Handle {
        let handle = self.next_handle;
        self.next_handle += 1;
        if self.next_handle == 0 { panic!("Handle overflow!"); } 
        handle
    }
}

static HOST_STATE: Lazy<Mutex<HostState>> = Lazy::new(|| Mutex::new(HostState::new()));

// This struct will implement the exported interface functions.
//
struct Component;

impl Guest for Component {
    fn allocate_buffer(size: u64) -> Result<Handle, HostError> {
        println!("[Provider Wasm] Allocating buffer of size {}", size);
        if size == 0 {
            return Err(HostError::Other("Cannot allocate zero-size buffer".to_string()));
        }
        let mut state = HOST_STATE.lock().unwrap();
        let handle = state.new_handle();
        state.buffers.insert(handle, vec![0u8; size as usize]);
        Ok(handle)
    }

    fn free_buffer(h: Handle) -> Result<(), HostError> {
        println!("[Provider Wasm] Freeing buffer {}", h);
        let mut state = HOST_STATE.lock().unwrap();
        if state.buffers.remove(&h).is_some() {
            state.matrix_dims.remove(&h);
            Ok(())
        } else {
            Err(HostError::InvalidHandle)
        }
    }

    fn write_to_host(
        guest_bytes: Vec<u8>,
        target_handle: Handle,
        target_offset: u64,
    ) -> Result<(), HostError> {
        println!("[Provider Wasm] Writing {} bytes to handle {} at offset {}", guest_bytes.len(), target_handle, target_offset);
        let mut state = HOST_STATE.lock().unwrap();
        match state.buffers.get_mut(&target_handle) {
            Some(buffer) => {
                let offset = target_offset as usize;
                let end = offset + guest_bytes.len();
                if end > buffer.len() {
                    return Err(HostError::CopyOutOfBounds);
                }
                buffer[offset..end].copy_from_slice(&guest_bytes);
                Ok(())
            }
            None => Err(HostError::InvalidHandle),
        }
    }

    fn read_from_host(
        source_handle: Handle,
        source_offset: u64,
        len: u64,
    ) -> Result<Vec<u8>, HostError> {
        println!("[Provider Wasm] Reading {} bytes from handle {} at offset {}", len, source_handle, source_offset);
        let state = HOST_STATE.lock().unwrap();
        match state.buffers.get(&source_handle) {
            Some(buffer) => {
                let offset = source_offset as usize;
                let read_len = len as usize;
                if offset + read_len > buffer.len() {
                    return Err(HostError::CopyOutOfBounds);
                }
                Ok(buffer[offset..offset + read_len].to_vec())
            }
            None => Err(HostError::InvalidHandle),
        }
    }

    fn register_matrix_dimensions(h: Handle, dims: MatrixDimensions) -> Result<(), HostError> {
        println!("[Provider Wasm] Registering dimensions {}x{} for handle {}", dims.rows, dims.cols, h);
        let mut state = HOST_STATE.lock().unwrap();
        if !state.buffers.contains_key(&h) {
            return Err(HostError::InvalidHandle);
        }
        state.matrix_dims.insert(h, (dims.rows, dims.cols));
        Ok(())
    }


    fn matrix_multiply_f32(
        handle_a: Handle,
        handle_b: Handle,
    ) -> Result<Handle, HostError> {
        println!("[Provider Wasm] Matrix multiply f32 for A:{} and B:{}", handle_a, handle_b);
        let mut state = HOST_STATE.lock().unwrap();

        let (rows_a, cols_a) = *state.matrix_dims.get(&handle_a).ok_or(HostError::InvalidHandle)?;
        let buffer_a_bytes = state.buffers.get(&handle_a).ok_or(HostError::InvalidHandle)?;
        let matrix_a_data = bytes_to_f32_slice(buffer_a_bytes)
            .ok_or_else(|| HostError::Other("Failed to cast buffer A to f32".to_string()))?;
        if matrix_a_data.len() != (rows_a * cols_a) as usize { return Err(HostError::Other("Buffer A size mismatch with dims".to_string())); }
        let matrix_a = nalgebra::DMatrix::<f32>::from_row_slice(rows_a as usize, cols_a as usize, matrix_a_data);

        let (rows_b, cols_b) = *state.matrix_dims.get(&handle_b).ok_or(HostError::InvalidHandle)?;
        let buffer_b_bytes = state.buffers.get(&handle_b).ok_or(HostError::InvalidHandle)?;
        let matrix_b_data = bytes_to_f32_slice(buffer_b_bytes)
            .ok_or_else(|| HostError::Other("Failed to cast buffer B to f32".to_string()))?;
        if matrix_b_data.len() != (rows_b * cols_b) as usize { return Err(HostError::Other("Buffer B size mismatch with dims".to_string())); }
        let matrix_b = nalgebra::DMatrix::<f32>::from_row_slice(rows_b as usize, cols_b as usize, matrix_b_data);

        if cols_a != rows_b {
            return Err(HostError::DimensionMismatch);
        }

        let matrix_c = matrix_a * matrix_b;
        let handle_c = state.new_handle();
        let c_bytes = f32_slice_to_bytes(matrix_c.as_slice());
        state.buffers.insert(handle_c, c_bytes);
        state.matrix_dims.insert(handle_c, (matrix_c.nrows() as u32, matrix_c.ncols() as u32));
        println!("[Provider Wasm] Stored result C ({},{}) with handle {}", matrix_c.nrows(), matrix_c.ncols(), handle_c);
        Ok(handle_c)
    }

    fn get_matrix_dimensions(h: Handle) -> Result<MatrixDimensions, HostError> {
        println!("[Provider Wasm] Getting dimensions for handle {}", h);
        let state = HOST_STATE.lock().unwrap();
        match state.matrix_dims.get(&h) {
            Some(&(rows, cols)) => Ok(MatrixDimensions { rows, cols }),
            None => Err(HostError::InvalidHandle),
        }
    }
}


fn bytes_to_f32_slice(bytes: &[u8]) -> Option<&[f32]> {
    if bytes.as_ptr() as usize % std::mem::align_of::<f32>() != 0 { return None; } // Alignment check
    if bytes.len() % std::mem::size_of::<f32>() != 0 { return None; }
    unsafe {
        Some(std::slice::from_raw_parts(
            bytes.as_ptr() as *const f32,
            bytes.len() / std::mem::size_of::<f32>(),
        ))
    }
}

fn f32_slice_to_bytes(floats: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(floats.len() * std::mem::size_of::<f32>());
    for float_val in floats {
        bytes.extend_from_slice(&float_val.to_ne_bytes());
    }
    bytes
}


bindings::export!(Component with_types_in bindings); 

