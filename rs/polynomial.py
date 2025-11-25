# Copyright (c) 2010 Andrew Brown <brownan@cs.duke.edu, brownan@gmail.com>
# See LICENSE.txt for license terms

from io import StringIO

class Polynomial(object):
    """Completely general polynomial class.
    
    Polynomial objects are immutable.
    
    Implementation note: while this class is mostly agnostic to the type of
    coefficients used (as long as they support the usual mathematical
    operations), the Polynomial class still assumes the additive identity and
    multiplicative identity are 0 and 1 respectively. If you're doing math over
    some strange field or using non-numbers as coefficients, this class will
    need to be modified."""
    def __init__(self, coefficients=(), **sparse):
        """
        There are three ways to initialize a Polynomial object.
        1) With a list, tuple, or other iterable, creates a polynomial using
        the items as coefficients in order of decreasing power

        2) With keyword arguments such as for example x3=5, sets the
        coefficient of x^3 to be 5

        3) With no arguments, creates an empty polynomial, equivalent to
        Polynomial((0,))

        >>> print Polynomial((5, 0, 0, 0, 0, 0))
        5x^5

        >>> print Polynomial(x32=5, x64=8)
        8x^64 + 5x^32

        >>> print Polynomial(x5=5, x9=4, x0=2) 
        4x^9 + 5x^5 + 2
        """
        if coefficients and sparse:
            raise TypeError("Specify coefficients list /or/ keyword terms, not"
                    " both")
        if coefficients:
            # Polynomial((1, 2, 3, ...))
            c = list(coefficients)
            # Expunge any leading 0 coefficients
            while c and c[0] == 0:
                c.pop(0)
            if not c:
                c.append(0)

            self.coefficients = tuple(c)
        elif sparse:
            # Polynomial(x32=...)
            # Get the sparse keyword keys (like 'x32', 'x64') and sort them
            # in descending order so the highest power comes first.
            powers = sorted(sparse.keys(), reverse=True)
            # Not catching possible exceptions from the following line, let
            # them bubble up.
            highest = int(powers[0][1:])
            coefficients = [0] * (highest+1)

            for power, coeff in sparse.items():
                power = int(power[1:])
                coefficients[highest - power] = coeff

            self.coefficients = tuple(coefficients)
        else:
            # Polynomial()
            self.coefficients = (0,)

    def __len__(self):
        """Returns the number of terms in the polynomial"""
        return len(self.coefficients)
    def degree(self):
        """Returns the degree of the polynomial"""
        return len(self.coefficients) - 1

    def __add__(self, other):
        diff = len(self) - len(other)
        if diff > 0:
            t1 = self.coefficients
            t2 = (0,) * diff + other.coefficients
        else:
            t1 = (0,) * (-diff) + self.coefficients
            t2 = other.coefficients

        return self.__class__(x+y for x,y in zip(t1, t2))

    def __neg__(self):
        return self.__class__(-x for x in self.coefficients)
    def __sub__(self, other):
        return self + -other
            
    def __mul__(self, other):
        terms = [0] * (len(self) + len(other))

        for i1, c1 in enumerate(reversed(self.coefficients)):
            if c1 == 0:
                # Optimization
                continue
            for i2, c2 in enumerate(reversed(other.coefficients)):
                terms[i1+i2] += c1*c2

        return self.__class__(reversed(terms))

    def __floordiv__(self, other):
        return divmod(self, other)[0]
    def __mod__(self, other):
        return divmod(self, other)[1]

    def __divmod__(dividend, divisor):
        """Implements polynomial long-division recursively. I know this is
        horribly inefficient, no need to rub it in. I know it can even throw
        recursion depth errors on some versions of Python.

        However, not being a math person myself, I implemented this from my
        memory of how polynomial long division works. It's straightforward and
        doesn't do anything fancy. There's no magic here.
        """
        class_ = dividend.__class__

        # See how many times the highest order term
        # of the divisor can go into the highest order term of the dividend

        dividend_power = dividend.degree()
        dividend_coefficient = dividend.coefficients[0]

        divisor_power = divisor.degree()
        divisor_coefficient = divisor.coefficients[0]

        quotient_power = dividend_power - divisor_power
        if quotient_power < 0:
            # Doesn't divide at all, return 0 for the quotient and the entire
            # dividend as the remainder
            return class_((0,)), dividend

        # Compute how many times the highest order term in the divisor goes
        # into the dividend
        quotient_coefficient = dividend_coefficient / divisor_coefficient
        quotient = class_( (quotient_coefficient,) + (0,) * quotient_power )

        remander = dividend - quotient * divisor

        if remander.coefficients == (0,):
            # Goes in evenly with no remainder, we're done
            return quotient, remander

        # There was a remainder, see how many times the remainder goes into the
        # divisor
        morequotient, remander = divmod(remander, divisor)
        return quotient + morequotient, remander

    def __eq__(self, other):
        return self.coefficients == other.coefficients
    def __ne__(self, other):
        return self.coefficients != other.coefficients
    def __hash__(self):
        return hash(self.coefficients)

    def __repr__(self):
        n = self.__class__.__name__
        return "%s(%r)" % (n, self.coefficients)
    def __str__(self):        
        prefix = '(GF(2^8) coefficients) '

        parts = []
        l = len(self) - 1
        for i, c in enumerate(self.coefficients):
            if not c and i > 0:
                continue
            power = l - i
            # Coerce coefficient to simple int for display (works for GF elements)
            ci = int(c)
            if ci == 1 and power != 0:
                coef_str = ""
            else:
                coef_str = str(ci)

            if power > 1:
                parts.append(f"{coef_str}x^{power}")
            elif power == 1:
                parts.append(f"{coef_str}x")
            else:
                parts.append(f"{coef_str}")

        body = ' + '.join(parts) if parts else '0'
        return prefix + body

    def evaluate(self, x):
        "Evaluate this polynomial at value x, returning the result."
        # Holds the sum over each term in the polynomial
        c = 0

        # Holds the current power of x. This is multiplied by x after each term
        # in the polynomial is added up. Initialized to x^0 = 1
        p = 1

        for term in reversed(self.coefficients):
            c = c + term * p

            p = p * x

        return c

    def get_coefficient(self, degree):
        """Returns the coefficient of the specified term"""
        if degree > self.degree():
            return 0
        else:
            return self.coefficients[-(degree+1)]


def _gf_solve_linear_system(A, b):
    """Solve A x = b over GF(2^8) using Gaussian elimination.

    A: list of rows (each row a list of GF elements), shape (n_rows, n_cols)
    b: list of GF elements length n_rows

    Returns list x of length n_cols if a unique solution exists.
    Raises ValueError on inconsistent or underdetermined systems.
    """
    from ff import GF256int

    # Copy matrix to avoid mutating inputs
    n_rows = len(A)
    if n_rows == 0:
        return []
    n_cols = len(A[0])

    M = [list(row) + [bval] for row, bval in zip([list(r) for r in A], b)]

    row = 0
    for col in range(n_cols):
        # Find pivot in or below current row
        sel = None
        for r in range(row, n_rows):
            if M[r][col] != GF256int(0):
                sel = r
                break
        if sel is None:
            # no pivot in this column
            continue

        # Swap to put pivot on current row
        if sel != row:
            M[row], M[sel] = M[sel], M[row]

        # Normalize pivot to 1
        pivot = M[row][col]
        inv = pivot.inverse()
        M[row] = [val * inv for val in M[row]]

        # Eliminate other rows
        for r in range(n_rows):
            if r == row:
                continue
            factor = M[r][col]
            if factor == GF256int(0):
                continue
            M[r] = [rv - factor * pv for rv, pv in zip(M[r], M[row])]

        row += 1
        if row == n_rows:
            break

    # Check for inconsistency: any row with all-zero coefficients but non-zero rhs
    for r in range(n_rows):
        all_zero = True
        for c in range(n_cols):
            if M[r][c] != GF256int(0):
                all_zero = False
                break
        if all_zero and M[r][-1] != GF256int(0):
            raise ValueError("Inconsistent linear system")

    # Identify pivot columns and ensure full column rank
    pivots = {}
    for r in range(n_rows):
        for c in range(n_cols):
            if M[r][c] == GF256int(1):
                # ensure this is the leading 1 (no previous non-zero before c)
                leading = True
                for cc in range(0, c):
                    if M[r][cc] != GF256int(0):
                        leading = False
                        break
                if leading:
                    pivots[c] = r
                    break

    if len(pivots) < n_cols:
        raise ValueError("Underdetermined system or multiple solutions")

    # Construct solution vector
    x = [GF256int(0)] * n_cols
    for c, r in pivots.items():
        x[c] = M[r][-1]

    return x


def berlekamp_welch_decode(y_values, k, xs, alpha=3):
    """Decode using the Berlekamp–Welch algorithm.

    y_values: iterable of received symbols (ints 0..255 or GF256int)
    k: message length (degree of message polynomial < k)
    xs: iterable of evaluation points (GF256int).

    Returns: Polynomial P (message polynomial) of degree < k.
    Raises ValueError if system cannot be solved uniquely.
    """
    from ff import GF256int

    # Convert ys to GF elements
    ys = [GF256int(y) if not isinstance(y, GF256int) else y for y in y_values]
    n = len(ys)

    # t is error-correction capability
    t = (n - k) // 2
    print("n, k, t =", n, k, t)
    m_q = k + t
    m_e = t
    n_unknowns = m_q + m_e
    print("m_q, m_e, n_unknowns =", m_q, m_e, n_unknowns)
    print("n (equations) =", n)


    # Degrees/unknown counts
    # Q has degree <= k-1 + t  => m_q = k + t coefficients (q0..q_{m_q-1})
    m_q = k + t
    # E has degree <= t, but we fix leading coefficient e_t = 1, so unknowns e0..e_{t-1}
    m_e = t

    n_unknowns = m_q + m_e

    # Build linear system A u = rhs
    A = []
    rhs = []
    for xi, yi in zip(xs, ys):
        # q part: powers xi^0 .. xi^{m_q-1}
        row = []
        p = GF256int(1)
        for _ in range(m_q):
            row.append(p)
            p = p * xi

        # e part: coefficients for e0..e_{t-1} are (-yi * xi^b)
        p = GF256int(1)
        for _ in range(m_e):
            row.append( (GF256int(0) - (yi * p)) )
            p = p * xi

        A.append(row)
        # RHS is yi * xi^t (because we moved yi*e_t*xi^t to RHS and e_t=1)
        rhs.append( yi * (xi ** t) )

    # Solve
    u = _gf_solve_linear_system(A, rhs)

    from ff import GF256int as _G

    q_coeffs = u[:m_q]  # ascending powers
    e_coeffs = u[m_q:]  # e0..e_{t-1}
    # append leading 1 for e_t
    e_full = list(e_coeffs) + [GF256int(1)]

    # Build Polynomial objects (Polynomial expects coefficients in descending power)
    Q = Polynomial(reversed(q_coeffs))
    E = Polynomial(reversed(e_full))

    # Divide Q by E to recover P
    P = Q // E

    return P
