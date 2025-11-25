from typing import List, Tuple

def generate_hamming_matrices(r: int) -> Tuple[int, int, List[int], List[int]]:
    """Return (n, k, parity_positions, data_positions) for Hamming(2^r-1, 2^r-1-r).
    Data positions 1..k, parity positions k+1..n (1-indexed).
    """
    n = (1 << r) - 1
    k = n - r
    data_positions = [i for i in range(1, k + 1)]
    parity_positions = [i for i in range(k + 1, n + 1)]
    return n, k, parity_positions, data_positions

def int_to_bits(x: int, length: int) -> List[int]:
    return [((x >> i) & 1) for i in range(length - 1, -1, -1)]

def bits_to_int(bits: List[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | (b & 1)
    return v

def col_bits(pos: int, r: int) -> List[int]:
    """Return column vector for position pos as an r-bit list (MSB-first)."""
    return [((pos >> (r - 1 - j)) & 1) for j in range(r)]

def codeword_to_str(cw: List[int]) -> str:
    return ''.join(str(b) for b in cw)

def extract_data_from_codeword(codeword: List[int], r: int) -> List[int]:
    n, k, parity_positions, data_positions = generate_hamming_matrices(r)
    if len(codeword) != n:
        raise ValueError(f"codeword length must be {n}, got {len(codeword)}")
    return [codeword[pos - 1] for pos in data_positions]


# ---------- GF(2) matrix utilities ----------

def mat_mul(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """Multiply A (m x p) by B (p x n) over GF(2)."""
    m = len(A)
    p = len(A[0]) if A else 0
    # assume B has p rows
    n = len(B[0])
    C = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = 0
            for t in range(p):
                acc ^= (A[i][t] & B[t][j])
            C[i][j] = acc
    return C


def mat_transpose(A: List[List[int]]) -> List[List[int]]:
    if not A:
        return []
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def identity(n: int) -> List[List[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def gf2_inverse(square: List[List[int]]) -> List[List[int]]:
    """Invert an r x r matrix over GF(2). Raises ValueError if singular."""
    r = len(square)
    # build augmented matrix [A | I]
    M = [row[:] + identity(r)[i] for i, row in enumerate(square)]
    cols = 2 * r
    # Gaussian elimination
    row = 0
    for col in range(r):
        # find pivot
        pivot = None
        for i in range(row, r):
            if M[i][col] == 1:
                pivot = i
                break
        if pivot is None:
            raise ValueError("Matrix is singular over GF(2); cannot invert")
        if pivot != row:
            M[row], M[pivot] = M[pivot], M[row]
        # eliminate other rows
        for i in range(r):
            if i != row and M[i][col] == 1:
                # row_i ^= row_row
                for j in range(col, cols):
                    M[i][j] ^= M[row][j]
        row += 1
        if row == r:
            break
    # extract inverse (right half)
    inv = [row[r:] for row in M]
    return inv