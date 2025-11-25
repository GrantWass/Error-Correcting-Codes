from rs import RSCoder
from polynomial import Polynomial
from ff import GF256int

def berlekamp_welch_decode(received: str, k: int):
    """
    Corrected Berlekamp-Welch decoding.
    """
    n = len(received)
    
    # Use proper evaluation points (powers of primitive element)
    alpha = GF256int(3)  # primitive element
    x_vals = [alpha ** i for i in range(n)]
    y_vals = [GF256int(ord(c)) for c in received]

    print("\n--- Debug: Evaluation points (x_i, y_i) ---")
    for xi, yi in zip(x_vals, y_vals):
        print(f"x = {xi} (0x{int(xi):02x}), y = {yi} (0x{int(yi):02x})")

    # Try different error counts
    for t in range((n - k) // 2, 0, -1):
        print(f"\n--- Trying t = {t} errors ---")
        
        # Q(x) has degree ≤ k + t - 1 (k+t coefficients)
        # E(x) has degree exactly t (t+1 coefficients)
        num_Q = k + t
        num_E = t + 1
        
        # Total unknowns: (k+t) + (t+1) = k + 2t + 1
        # We have n equations from points + 1 normalization = n+1 equations
        # Need: n+1 ≥ k + 2t + 1 ⇒ n ≥ k + 2t (which is true for RS codes)
        
        equations = []
        # Point equations: Q(x_i) = y_i * E(x_i) for each point
        for xi, yi in zip(x_vals, y_vals):
            row = [xi ** j for j in range(num_Q)]           # Q coefficients
            row += [-yi * (xi ** j) for j in range(num_E)]  # E coefficients
            equations.append((row, GF256int(0)))
        
        # Normalization: highest coefficient of E(x) = 1
        # This goes in E's coefficients at position t (since E has t+1 coefficients)
        norm_row = [GF256int(0)] * num_Q + [GF256int(0)] * t + [GF256int(1)]
        equations.append((norm_row, GF256int(1)))
        
        print(f"System: {len(equations)} equations, {num_Q + num_E} unknowns")
        
        # Solve system
        try:
            # Convert to matrix form for Gaussian elimination
            matrix_eqs = [list(row) + [rhs] for row, rhs in equations]
            
            # Gaussian elimination
            n_rows = len(matrix_eqs)
            n_cols = len(matrix_eqs[0])
            
            for col in range(num_Q + num_E):
                # Find pivot
                pivot_row = None
                for r in range(col, n_rows):
                    if matrix_eqs[r][col] != GF256int(0):
                        pivot_row = r
                        break
                
                if pivot_row is None:
                    print(f"No pivot found for column {col}, system may be singular")
                    break
                
                # Swap pivot row to current position
                matrix_eqs[col], matrix_eqs[pivot_row] = matrix_eqs[pivot_row], matrix_eqs[col]
                
                # Normalize pivot row
                pivot_val = matrix_eqs[col][col]
                matrix_eqs[col] = [x / pivot_val for x in matrix_eqs[col]]
                
                # Eliminate below and above
                for r in range(n_rows):
                    if r != col and matrix_eqs[r][col] != GF256int(0):
                        factor = matrix_eqs[r][col]
                        matrix_eqs[r] = [matrix_eqs[r][c] - factor * matrix_eqs[col][c] 
                                       for c in range(n_cols)]
            
            # Extract solution
            solution = [matrix_eqs[i][-1] for i in range(num_Q + num_E)]
            
            Q_coeffs = solution[:num_Q]
            E_coeffs = solution[num_Q:]
            
            Q = Polynomial(Q_coeffs)
            E = Polynomial(E_coeffs)
            
            print(f"Q(x) = {Q}")
            print(f"E(x) = {E}")
            
            # Check if E(x) divides Q(x)
            try:
                M, remainder = Q.__divmod__(E)
                if remainder == Polynomial([GF256int(0)]):
                    # Success!
                    decoded = "".join(chr(int(c)) for c in M.coefficients[:k])
                    print(f"Success! Decoded message: {repr(decoded)}")
                    return decoded
                else:
                    print(f"Remainder not zero: {remainder}")
            except ZeroDivisionError:
                print("E(x) is zero polynomial")
                
        except Exception as e:
            print(f"Solution failed for t={t}: {e}")
            continue
    
    raise Exception("Berlekamp-Welch decoding failed")

# Also fix your test case - make sure you're introducing errors properly
print('\nCorrected example:')
coder = RSCoder(7, 4)
c_poly2 = coder.encode("0001", poly=True)
c2 = "".join(chr(int(x)) for x in c_poly2.coefficients).rjust(coder.n, "\0")

print('Original:', repr("0001"))
print('Encoded:', repr(c2))

# Introduce error by flipping first symbol
r2 = chr(ord(c2[0]) ^ 1) + c2[1:]  # XOR with 1 to flip a bit
print('Received:', repr(r2))

result2 = berlekamp_welch_decode(r2, coder.k)
print('Final result:', repr(result2))