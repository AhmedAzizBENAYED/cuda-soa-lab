#!/usr/bin/env python3
"""
CUDA Sanity Test Script
Tests basic CUDA functionality to ensure GPU is accessible
"""

from numba import cuda
import numpy as np
import sys


@cuda.jit
def simple_add_kernel(x, out):
    """Simple CUDA kernel that adds 10 to each element"""
    idx = cuda.grid(1)
    if idx < x.size:
        out[idx] = x[idx] + 10


def test_cuda_availability():
    """Test if CUDA is available"""
    print("=" * 60)
    print("🧪 CUDA Availability Test")
    print("=" * 60)

    if not cuda.is_available():
        print("❌ CUDA is NOT available!")
        print("   This system does not have CUDA support.")
        return False

    print("✅ CUDA is available")
    return True


def test_gpu_devices():
    """Test GPU device detection"""
    print("\n" + "=" * 60)
    print("🎮 GPU Device Detection")
    print("=" * 60)

    try:
        gpus = cuda.gpus
        if len(gpus) == 0:
            print("❌ No GPU devices detected!")
            return False

        print(f"✅ Found {len(gpus)} GPU device(s):")
        for gpu in gpus:
            print(f"   - {gpu.name}")
            print(f"     Compute Capability: {gpu.compute_capability}")

        return True
    except Exception as e:
        print(f"❌ Error detecting GPU devices: {e}")
        return False


def test_simple_kernel():
    """Test a simple CUDA kernel execution"""
    print("\n" + "=" * 60)
    print("⚡ Simple Kernel Execution Test")
    print("=" * 60)

    try:
        # Create test data
        size = 100
        x = np.arange(size, dtype=np.float32)
        expected = x + 10

        # Allocate device memory
        d_x = cuda.to_device(x)
        d_out = cuda.device_array(size, dtype=np.float32)

        # Launch kernel
        threads_per_block = 32
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

        print(f"   Launching kernel with:")
        print(f"   - Grid size: {blocks_per_grid} blocks")
        print(f"   - Block size: {threads_per_block} threads")
        print(f"   - Total threads: {blocks_per_grid * threads_per_block}")

        simple_add_kernel[blocks_per_grid, threads_per_block](d_x, d_out)

        # Copy result back
        result = d_out.copy_to_host()

        # Verify results
        if np.allclose(result, expected):
            print("✅ Kernel execution successful!")
            print(f"   Input sample: {x[:5]}")
            print(f"   Output sample: {result[:5]}")
            return True
        else:
            print("❌ Kernel execution failed - incorrect results!")
            print(f"   Expected: {expected[:5]}")
            print(f"   Got: {result[:5]}")
            return False

    except Exception as e:
        print(f"❌ Kernel execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2d_kernel():
    """Test 2D kernel for matrix operations"""
    print("\n" + "=" * 60)
    print("🔲 2D Matrix Kernel Test")
    print("=" * 60)

    try:
        @cuda.jit
        def matrix_multiply_by_two(matrix, out):
            i, j = cuda.grid(2)
            if i < matrix.shape[0] and j < matrix.shape[1]:
                out[i, j] = matrix[i, j] * 2

        # Create small test matrix
        rows, cols = 10, 10
        matrix = np.random.rand(rows, cols).astype(np.float32)
        expected = matrix * 2

        # Transfer to device
        d_matrix = cuda.to_device(matrix)
        d_out = cuda.device_array_like(d_matrix)

        # Launch 2D kernel
        threads_per_block = (4, 4)
        blocks_per_grid_x = (rows + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (cols + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        print(f"   Matrix size: {rows}x{cols}")
        print(f"   Block dimensions: {threads_per_block}")
        print(f"   Grid dimensions: {blocks_per_grid}")

        matrix_multiply_by_two[blocks_per_grid, threads_per_block](d_matrix, d_out)

        result = d_out.copy_to_host()

        if np.allclose(result, expected):
            print("✅ 2D kernel execution successful!")
            return True
        else:
            print("❌ 2D kernel execution failed!")
            return False

    except Exception as e:
        print(f"❌ 2D kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all CUDA tests"""
    print("\n" + "🚀 " + "=" * 56 + " 🚀")
    print("   CUDA SANITY CHECK - GPU SERVICE DEPLOYMENT")
    print("🚀 " + "=" * 56 + " 🚀\n")

    tests = [
        ("CUDA Availability", test_cuda_availability),
        ("GPU Device Detection", test_gpu_devices),
        ("Simple 1D Kernel", test_simple_kernel),
        ("2D Matrix Kernel", test_2d_kernel),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print("\n" + "-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("-" * 60)

    if passed == total:
        print("\n🎉 All tests passed! GPU is ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check your CUDA setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())