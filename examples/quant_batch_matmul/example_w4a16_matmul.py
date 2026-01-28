"""
W4A16 Quantized MatMul Example for Ascend NPU 910B

This example demonstrates a W4A16 (4-bit weights, 16-bit activations) quantized matrix multiplication
kernel for Ascend NPU 910B.

The workflow:
1. Create a weight matrix in fp16 but restricted to int4 range [-8, 7] (inclusive)
2. Create an activation matrix in fp16 (random)
3. Pack the int4 weights into int8 storage (2 int4 values per int8)
4. Ascend NPU Kernel:
   - Load Activation Matrix onto Cube memory (L1)
   - Load packed Weight matrix into vector memory (UB)
   - Dequantize weights from packed int4 back to fp16 on vector core
   - Transfer dequantized weights to L1 for cube core
   - Perform matmul with float32 accumulation
   - Output C in float16
"""

import argparse
from typing import Literal

import tilelang as tl
import tilelang.language as T
import torch

# Constants for int4 range
INT4_MIN = -8
INT4_MAX = 7


def pack_int4_to_int8(weight_int4: torch.Tensor) -> torch.Tensor:
    """
    Pack two int4 values into one int8 value.

    Args:
        weight_int4: Weight tensor with int4 values in int8 dtype, shape [..., N]
                    N must be even (we pack pairs of values)

    Returns:
        Packed tensor with shape [..., N//2] in int8 dtype
    """
    assert weight_int4.shape[-1] % 2 == 0, "Last dimension must be even for int4 packing"

    # Reshape to pair adjacent elements
    shape = weight_int4.shape
    weight_pairs = weight_int4.view(*shape[:-1], shape[-1] // 2, 2)

    # Convert to unsigned for packing (shift from [-8,7] to [0,15])
    weight_unsigned = (weight_pairs + 8).to(torch.uint8)

    # Pack: low 4 bits from first element, high 4 bits from second element
    packed = (weight_unsigned[..., 0] & 0x0F) | ((weight_unsigned[..., 1] & 0x0F) << 4)

    return packed.to(torch.int8)


def unpack_int8_to_int4_fp16(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack int8 to two int4 values and convert to fp16.

    Args:
        packed: Packed tensor with shape [..., N] in int8 dtype

    Returns:
        Unpacked tensor with shape [..., N*2] in fp16 dtype
    """
    # Extract low and high 4 bits
    packed_uint = packed.to(torch.uint8)
    low = (packed_uint & 0x0F).to(torch.int8)
    high = ((packed_uint >> 4) & 0x0F).to(torch.int8)

    # Convert back from unsigned [0,15] to signed [-8,7]
    low = low.to(torch.int16) - 8
    high = high.to(torch.int16) - 8

    # Interleave and convert to fp16
    shape = packed.shape
    unpacked = torch.stack([low, high], dim=-1).view(*shape[:-1], shape[-1] * 2)

    return unpacked.to(torch.float16)


@tl.jit(
    out_idx=[2],
    workspace_idx=[3],
    pass_configs={
        tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }
)
def w4a16_matmul(
    M: int, N: int, K: int,
    block_M: int, block_N: int, block_K: int,
    dtype: Literal["float16"] = "float16",
    accum_dtype: Literal["float32", "float"] = "float",
):
    """
    W4A16 Quantized MatMul kernel for Ascend NPU 910B.

    The kernel performs A @ B where:
    - A: Activation matrix in float16 [M, K]
    - B: Weight matrix stored as packed int4 in int8 [K, N//2]

    Inside the kernel:
    1. Load A into L1 (cube memory)
    2. Load packed B into UB (vector memory)
    3. Dequantize B from int4 to fp16 on vector core
    4. Copy dequantized B to L1
    5. Perform GEMM on cube core with float32 accumulation
    6. Output C in float16
    """

    VEC_NUM = 2  # Number of vector units per cube core
    CAST_MODE = "CAST_RINT"

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)

    # Packed dimension (2 int4 values per int8)
    N_packed = N // 2
    block_N_packed = block_N // 2
    block_M_2 = T.ceildiv(block_M, VEC_NUM)

    @T.prim_func
    def main(
            A: T.Tensor([M, K], dtype),              # Activation matrix in fp16
            B_packed: T.Tensor([K, N_packed], "int8"),  # Packed int4 weights
            C: T.Tensor([M, N], dtype),              # Output matrix in fp16
            workspace: T.Tensor([M, N], accum_dtype),  # Intermediate buffer for L0C -> UB transfer
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            # Calculate block indices
            bm = cid // n_num
            bn = cid % n_num

            # L1 buffers for cube core operations
            A_L1 = T.alloc_L1([block_M, block_K], dtype)
            B_L1 = T.alloc_L1([block_K, block_N], dtype)

            # L0C accumulator for GEMM result
            C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

            # UB (Unified Buffer) for vector operations - dequantization
            B_packed_ub = T.alloc_ub([block_K, block_N_packed], "int8")
            B_unpacked_ub = T.alloc_ub([block_K, block_N], dtype)

            # UB for output processing
            c_ub = T.alloc_ub([block_M_2, block_N], accum_dtype)
            c_out = T.alloc_ub([block_M_2, block_N], dtype)

            # K-loop: iterate over K dimension in blocks
            for bk in T.serial(k_num):
                # Step 1: Load activation tile A to L1 (cube memory)
                T.copy(A[bm * block_M, bk * block_K], A_L1)

                # Step 2: Load packed weights to UB (vector memory)
                T.copy(B_packed[bk * block_K, bn * block_N_packed], B_packed_ub)

                # Step 3: Dequantize int4 weights to fp16 on vector core
                # Unpack: each int8 contains 2 int4 values
                # Low 4 bits -> first value, high 4 bits -> second value
                for ki, ni in T.Parallel(block_K, block_N_packed):
                    # Extract low 4 bits and convert to fp16 (shift from [0,15] to [-8,7])
                    low_val = ((B_packed_ub[ki, ni].astype("int16") & 0x0F) - 8).astype(dtype)
                    # Extract high 4 bits and convert to fp16
                    high_val = (((B_packed_ub[ki, ni].astype("int16") >> 4) & 0x0F) - 8).astype(dtype)
                    # Store unpacked values
                    B_unpacked_ub[ki, ni * 2] = low_val
                    B_unpacked_ub[ki, ni * 2 + 1] = high_val

                # Step 4: Copy dequantized weights from UB to L1 (cube memory)
                T.copy(B_unpacked_ub, B_L1)

                # Step 5: Perform GEMM on cube core with float32 accumulation
                T.gemm_v0(A_L1, B_L1, C_L0, init=(bk == 0))

            # Step 6: Copy L0C result to workspace (global memory)
            T.copy(C_L0, workspace[bm * block_M, bn * block_N])

            # Step 7: Each vector unit loads its portion from workspace to UB
            T.copy(workspace[bm * block_M + vid * block_M_2, bn * block_N], c_ub)

            # Step 8: Cast from accum_dtype (float32) to output dtype (float16)
            T.tile.cast(c_out, c_ub, mode=CAST_MODE, count=block_M_2 * block_N)

            # Step 9: Write output to global memory
            T.copy(c_out, C[bm * block_M + vid * block_M_2, bn * block_N])

    return main


def ref_program(A: torch.Tensor, B_packed: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation for W4A16 matmul.

    Args:
        A: Activation matrix [M, K] in fp16
        B_packed: Packed weight matrix [K, N//2] in int8

    Returns:
        Output matrix [M, N] in fp16
    """
    # Unpack weights to fp16
    B_unpacked = unpack_int8_to_int4_fp16(B_packed)
    # Perform matmul in float32 and cast back to fp16
    return (A.float() @ B_unpacked.float()).half()


def check_case(
    M: int, N: int, K: int,
    block_M: int, block_N: int, block_K: int,
    dtype: Literal["float16"] = "float16",
    accum_dtype: Literal["float32", "float"] = "float",
):
    """
    Test case for W4A16 matmul.

    Requirements:
    - M must be divisible by block_M
    - N must be even (for int4 packing) and divisible by block_N
    - K must be divisible by block_K
    """
    # Validate dimension requirements
    assert N % 2 == 0, "N must be even for int4 packing (2 int4 values per int8)"
    assert M % block_M == 0, f"M ({M}) must be divisible by block_M ({block_M})"
    assert N % block_N == 0, f"N ({N}) must be divisible by block_N ({block_N})"
    assert K % block_K == 0, f"K ({K}) must be divisible by block_K ({block_K})"

    # Create activation matrix in fp16 (random values)
    A = torch.randn([M, K], dtype=torch.float16)

    # Create weight matrix with int4 values (-8 to 7) stored in int8
    # Using randint to generate values in int4 range
    B_int4 = torch.randint(INT4_MIN, INT4_MAX + 1, [K, N], dtype=torch.int8)

    # Pack int4 values into int8 (2 values per byte)
    B_packed = pack_int4_to_int8(B_int4)

    # Create kernel
    kernel = w4a16_matmul(M, N, K, block_M, block_N, block_K, dtype, accum_dtype)

    # Run kernel on NPU
    C = kernel(A.npu(), B_packed.npu())

    # Compute reference on CPU
    ref_C = ref_program(A, B_packed)

    # Compare results
    torch.testing.assert_close(C.cpu(), ref_C, rtol=1e-2, atol=1e-2)


def main(custom_args=None):
    parser = argparse.ArgumentParser(description="W4A16 QuantMatmul Example for Ascend NPU 910B")
    parser.add_argument("--m", type=int, default=1024, help="Matrix M dimension")
    parser.add_argument("--n", type=int, default=1024, help="Matrix N dimension")
    parser.add_argument("--k", type=int, default=1024, help="Matrix K dimension")
    args, remains = parser.parse_known_args(custom_args)
    if remains:
        print(f"[{parser.description}]", "Unknown args:", remains)

    M, N, K = args.m, args.n, args.k

    tl.cache.clear_cache()
    torch.manual_seed(0)

    # Test case with standard block sizes
    check_case(M, N, K, block_M=128, block_N=256, block_K=64)

    print("W4A16 QuantMatmul example passed!")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()

