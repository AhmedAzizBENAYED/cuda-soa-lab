from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
import numpy as np
from numba import cuda
import io
import time
import subprocess
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# ============================================
# Prometheus Metrics
# ============================================
REQUEST_COUNT = Counter('gpu_service_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('gpu_service_request_latency_seconds', 'Request latency', ['endpoint'])
GPU_MEMORY_USED = Gauge('gpu_memory_used_mb', 'GPU memory used in MB', ['gpu_id'])
GPU_MEMORY_TOTAL = Gauge('gpu_memory_total_mb', 'GPU memory total in MB', ['gpu_id'])

# ============================================
# FastAPI App
# ============================================
app = FastAPI(title="GPU Matrix Addition Service", version="1.0")

# TODO: Change this to your assigned port!
STUDENT_PORT = 8110


# ============================================
# CUDA Kernel for Matrix Addition
# ============================================
@cuda.jit
def matrix_add_kernel(A, B, C):
    """
    GPU kernel for matrix addition.
    Each thread computes one element: C[i,j] = A[i,j] + B[i,j]

    How it works:
    - cuda.grid(2) returns the global (i, j) position of this thread
    - Each thread handles ONE matrix element
    - Bounds checking ensures we don't access out-of-range memory
    """
    # Get 2D thread indices
    i, j = cuda.grid(2)

    # Bounds checking
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = A[i, j] + B[i, j]


def gpu_matrix_add(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple:
    """
    Perform matrix addition on GPU using Numba CUDA.

    Args:
        matrix_a: First input matrix (numpy array)
        matrix_b: Second input matrix (numpy array)

    Returns:
        Tuple of (result_matrix, elapsed_time)
    """
    # Validate shapes
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(f"Matrix shapes don't match: {matrix_a.shape} vs {matrix_b.shape}")

    # Ensure float32 for GPU compatibility
    matrix_a = matrix_a.astype(np.float32)
    matrix_b = matrix_b.astype(np.float32)

    # Start timing
    start_time = time.perf_counter()

    # Transfer data to GPU (Host -> Device)
    d_a = cuda.to_device(matrix_a)
    d_b = cuda.to_device(matrix_b)
    d_c = cuda.device_array_like(d_a)

    # Configure kernel launch parameters
    # Use 16x16 threads per block (common choice, 256 threads total)
    threads_per_block = (16, 16)

    # Calculate grid dimensions to cover entire matrix
    blocks_per_grid_x = (matrix_a.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (matrix_a.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch kernel: kernel[grid_size, block_size](arguments)
    matrix_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy result back to host (Device -> Host)
    result = d_c.copy_to_host()

    # End timing
    elapsed_time = time.perf_counter() - start_time

    return result, elapsed_time


# ============================================
# Helper Functions
# ============================================
def load_npz_matrix(file_content: bytes) -> np.ndarray:
    """Load matrix from uploaded .npz file"""
    try:
        npz_file = np.load(io.BytesIO(file_content))
        # Get the first array from the npz file
        array_name = npz_file.files[0]
        matrix = npz_file[array_name]
        return matrix
    except Exception as e:
        raise ValueError(f"Failed to load .npz file: {str(e)}")


def parse_nvidia_smi():
    """Parse nvidia-smi output to get GPU memory info"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 3:
                    gpus.append({
                        "gpu": parts[0],
                        "memory_used_MB": int(parts[1]),
                        "memory_total_MB": int(parts[2])
                    })
        return gpus
    except Exception as e:
        raise RuntimeError(f"Failed to query GPU info: {str(e)}")


# ============================================
# API Endpoints
# ============================================
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(endpoint='/health', status='success').inc()
    return {"status": "ok"}


@app.post("/add")
async def add_matrices(
        file_a: UploadFile = File(..., description="First matrix (.npz file)"),
        file_b: UploadFile = File(..., description="Second matrix (.npz file)")
):
    """
    Add two matrices on GPU.

    Accepts two .npz files containing numpy arrays and returns the computation time.
    """
    try:
        with REQUEST_LATENCY.labels(endpoint='/add').time():
            # Validate file extensions
            if not file_a.filename.endswith('.npz') or not file_b.filename.endswith('.npz'):
                REQUEST_COUNT.labels(endpoint='/add', status='error').inc()
                raise HTTPException(status_code=400, detail="Both files must be .npz format")

            # Read file contents
            content_a = await file_a.read()
            content_b = await file_b.read()

            # Load matrices
            matrix_a = load_npz_matrix(content_a)
            matrix_b = load_npz_matrix(content_b)

            # Validate shapes match
            if matrix_a.shape != matrix_b.shape:
                REQUEST_COUNT.labels(endpoint='/add', status='error').inc()
                raise HTTPException(
                    status_code=400,
                    detail=f"Matrix shapes don't match: {matrix_a.shape} vs {matrix_b.shape}"
                )

            # Perform GPU addition
            result, elapsed_time = gpu_matrix_add(matrix_a, matrix_b)

            REQUEST_COUNT.labels(endpoint='/add', status='success').inc()

            return {
                "matrix_shape": list(matrix_a.shape),
                "elapsed_time": round(elapsed_time, 6),
                "device": "GPU"
            }

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/add', status='error').inc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/gpu-info")
async def get_gpu_info():
    """
    Get GPU memory information from nvidia-smi.

    Returns GPU index, memory used, and total memory for all available GPUs.
    """
    try:
        gpus = parse_nvidia_smi()

        # Update Prometheus metrics
        for gpu in gpus:
            GPU_MEMORY_USED.labels(gpu_id=gpu["gpu"]).set(gpu["memory_used_MB"])
            GPU_MEMORY_TOTAL.labels(gpu_id=gpu["gpu"]).set(gpu["memory_total_MB"])

        REQUEST_COUNT.labels(endpoint='/gpu-info', status='success').inc()
        return {"gpus": gpus}

    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/gpu-info', status='error').inc()
        raise HTTPException(status_code=500, detail=f"Failed to get GPU info: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


# ============================================
# Main Entry Point
# ============================================
if __name__ == "__main__":
    import uvicorn

    # Check if CUDA is available
    if not cuda.is_available():
        print("⚠️  WARNING: CUDA is not available! This service requires a GPU.")
    else:
        print(f"✅ CUDA is available")
        print(f"🎮 GPU Devices: {cuda.gpus}")

    print(f"🚀 Starting GPU Matrix Addition Service on port {STUDENT_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=STUDENT_PORT)