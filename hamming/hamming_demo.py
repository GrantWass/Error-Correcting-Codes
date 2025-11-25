"""Demonstration of Hamming code examples: parity checks, coset decoding, nearest-neighbor decoding."""
import random

from ham import (
    encode_with_G,
    syndrome,
    coset_decode,
    generate_generator_matrix,
    encode_with_G,
    nearest_neighbor_decode
)

from code_utils import (
    codeword_to_str,
    generate_hamming_matrices,
)

def demo_r3():
    r = 3
    n, k, _, _ = generate_hamming_matrices(r)
    print(f'Hamming({n},{k})')
    G, _, _ = generate_generator_matrix(r)

    # random data
    data = [random.randint(0, 1) for _ in range(k)]

    # data = [1, 0, 1, 1]
    print('\nOriginal data bits:', data)
    code = encode_with_G(data, G)
    
    print("Generator matrix (G):")
    for row in G:
        print(row)
    print('Encoded codeword:    ', codeword_to_str(code))

    s_int, s_bits = syndrome(code, r)
    print('Syndrome (clean):    ', s_int, s_bits)

    # Single-bit error
    rx = code.copy()
    err_pos = 3  # 1-indexed position 3 (but list index 2)
    rx[err_pos - 1] ^= 1
    print(f"\nReceived with single-bit error at position {err_pos}:", codeword_to_str(rx))
    s_int, s_bits = syndrome(rx, r)
    print('Syndrome (received): ', s_int, s_bits)

    corrected, data_out = coset_decode(rx, r)
    print('Coset decode corrected:', codeword_to_str(corrected), '-> data', data_out)

    # Verify nearest-neighbor also finds the closest codeword
    nn_code, nn_data = nearest_neighbor_decode(rx, r)
    print('Nearest-neighbor:     ', codeword_to_str(nn_code), '-> data', nn_data)

    # Double-bit error example (may be uncorrectable by coset single-error decoder)
    rx2 = code.copy()
    rx2[1] ^= 1
    rx2[2] ^= 1
    print('\nReceived with double-bit error at positions 2 and 3:', codeword_to_str(rx2))
    s_int2, s_bits2 = syndrome(rx2, r)
    print('Syndrome (double):    ', s_int2, s_bits2)
    corr2, data2 = coset_decode(rx2, r)
    print('Coset decode (attempt):', codeword_to_str(corr2), '-> data', data2)
    nn_code2, nn_data2 = nearest_neighbor_decode(rx2, r)
    print('Nearest-neighbor best:', codeword_to_str(nn_code2), '-> data', nn_data2)


if __name__ == '__main__':
    demo_r3()
