#!/usr/bin/env python3
"""Create test matrices for GPU service testing"""

import numpy as np
import sys
import os


def create_matrix_pair(size=512, output_dir='.'):
    """Create a pair of random matrices and save as .npz files"""
    print(f"Creating {size}x{size} test matrices...")

    # Create random matrices
    matrix_a = np.random.rand(size, size).astype(np.float32)
    matrix_b = np.random.rand(size, size).astype(np.float32)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save as .npz files
    file_a = os.path.join(output_dir, f'matrix_a_{size}x{size}.npz')
    file_b = os.path.join(output_dir, f'matrix_b_{size}x{size}.npz')

    np.savez(file_a, arr_0=matrix_a)
    np.savez(file_b, arr_0=matrix_b)

    print(f"✅ Created: {file_a} ({os.path.getsize(file_a) / 1024:.2f} KB)")
    print(f"✅ Created: {file_b} ({os.path.getsize(file_b) / 1024:.2f} KB)")

    return file_a, file_b


def main():
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    create_matrix_pair(size, output_dir)


if __name__ == "__main__":
    main()