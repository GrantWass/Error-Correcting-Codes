from typing import List, Tuple
from code_utils import (bits_to_int, col_bits, mat_mul, mat_transpose,
    gf2_inverse, extract_data_from_codeword, generate_hamming_matrices, int_to_bits)

# ---------- Generator matrix construction ----------

def generate_generator_matrix(r:int) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Generate the systematic generator matrix G for a Hamming code with r parity bits.

    Returns:
        G (List[List[int]]): k x n generator matrix
        parity_positions (List[int])
        data_positions (List[int])
    """
    n, k, parity_positions, data_positions = generate_hamming_matrices(r)

    # Build H_d (r x k) and H_p (r x r)
    H_d = [[0] * k for _ in range(r)]
    H_p = [[0] * r for _ in range(r)]
    for col_idx, pos in enumerate(data_positions):
        bits = col_bits(pos, r)
        for row in range(r):
            H_d[row][col_idx] = bits[row]
    for col_idx, pos in enumerate(parity_positions):
        bits = col_bits(pos, r)
        for row in range(r):
            H_p[row][col_idx] = bits[row]

    # Compute S = H_p^{-1} * H_d  (r x k)
    H_p_inv = gf2_inverse(H_p)
    S = mat_mul(H_p_inv, H_d)  # r x k

    # generator parity block = (S)^T (k x r)
    parity_block = mat_transpose(S)  # k x r

    # Build G = [I_k | parity_block]
    G = []
    for i in range(k):
        row = [0] * k
        row[i] = 1
        row += parity_block[i][:]
        G.append(row)

    return G, parity_positions, data_positions


def encode_with_G(data_bits: List[int], G: List[List[int]]) -> List[int]:
    """
    Encode data bits using generator matrix G (systematic form).

    Returns:
        codeword (List[int]): encoded bits of length n
    """
    k = len(G)
    if k == 0:
        return []
    n = len(G[0])
    if len(data_bits) != k:
        raise ValueError(f"data_bits length must be {k}, got {len(data_bits)}")

    codeword = [0] * n
    for j in range(n):
        s = 0
        for i in range(k):
            s ^= (data_bits[i] & G[i][j])
        codeword[j] = s
    return codeword


# ---------- Syndrome + decoding ----------

def syndrome(received: List[int], r: int) -> Tuple[int, List[int]]:
    """
    Compute the syndrome vector and its integer value for a received Hamming codeword.

    Normally, this would be done via: s = H * r^T (mod 2),
    where H is the parity-check matrix (r x n).

    In this implementation, we don't explicitly construct H.
    Instead, we use the fact that in a standard Hamming(2^r-1, 2^r-1-r) code,
    each column of H is just the binary representation of its 1-indexed position.

    For example, for r=3 (Hamming(7,4)), H columns are:
        1 → 001
        2 → 010
        3 → 011
        4 → 100
        5 → 101
        6 → 110
        7 → 111
    """
    n, k, parity_positions, data_positions = generate_hamming_matrices(r)
    if len(received) != n:
        raise ValueError(f"received length must be {n}, got {len(received)}")

    bits = []
    for i in range(r - 1, -1, -1):
        s = 0
        for pos in range(1, n + 1):
            if ((pos >> i) & 1) != 0:
                s ^= (received[pos - 1] & 1)
        bits.append(s)
    return bits_to_int(bits), bits

def coset_decode(received: List[int], r: int) -> Tuple[List[int], List[int]]:
    n, k, parity_positions, data_positions = generate_hamming_matrices(r)
    s_int, _ = syndrome(received, r)
    corrected = received.copy()
    if s_int != 0 and 1 <= s_int <= n:
        corrected[s_int - 1] ^= 1
    data = extract_data_from_codeword(corrected, r)
    return corrected, data

def nearest_neighbor_decode(received: List[int], r: int) -> Tuple[List[int], List[int]]:
    n, k, parity_positions, data_positions = generate_hamming_matrices(r)
    min_distance = n + 1
    best_codeword = None
    G = generate_generator_matrix(r)[0]

    # Generate all codewords by encoding all possible data bits
    for data_int in range(1 << k):
        data_bits = int_to_bits(data_int, k)
        codeword = encode_with_G(data_bits, G)
        distance = sum(1 for i in range(n) if codeword[i] != received[i])
        if distance < min_distance:
            min_distance = distance
            best_codeword = codeword

    data = extract_data_from_codeword(best_codeword, r)
    return best_codeword, data