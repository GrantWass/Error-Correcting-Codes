"""Textual visualizer for Reed-Solomon encoding and decoding steps.

Run from the repository root with:
  python3 -m rs.visualize_rs

This prints step-by-step details for encoding and decoding using the
RSCoder implementation in this repo.
"""
from rs import RSCoder
from polynomial import Polynomial
from ff import GF256int
import matplotlib.pyplot as plt

def bits_of_bytes(s: str):
    return ' '.join(f"{ord(c):08b}" for c in s)


def visualize_encode(coder: RSCoder, message: str):
    print('\n=== ENCODING ===')
    print('Parameters: n=%d, k=%d, nsym (parity bits)=%d' % (coder.n, coder.k, coder.n - coder.k))
    print('Message (bytes):', repr(message))
    print('Message bits:     ', bits_of_bytes(message))

    # message polynomial
    m = Polynomial(GF256int(ord(x)) for x in message)
    print('\nMessage polynomial m(x):', str(m))

    # shifted
    shift = coder.n - coder.k
    mprime = m * Polynomial((GF256int(1),) + (GF256int(0),) * shift)
    print(f'm(x) * x^{shift}:', str(mprime))

    # division
    q, remainder = divmod(mprime, coder.g)
    # print the generator polynomial used
    print('\nGenerator polynomial g(x):', str(coder.g))
    print('\nDivide m(x)*x^nsym by g(x):')
    print('Quotient q(x):', str(q))
    print('Remainder r(x):', str(remainder))

    # codeword
    c = mprime - remainder
    print('\nEncoded codeword polynomial c(x):', str(c))
    # bytes
    cw_bytes = "".join(chr(int(x)) for x in c.coefficients).rjust(coder.n, "\0")
    print('Encoded bytes (repr):', repr(cw_bytes))
    print('Encoded bits:', bits_of_bytes(cw_bytes))
    return c, cw_bytes


def visualize_decode(coder: RSCoder, received: str):
    print('\n=== DECODING ===')
    print('Received bytes repr:', repr(received))
    print('Received bits:', bits_of_bytes(received))
    r_poly = Polynomial(GF256int(ord(x)) for x in received)
    print('Received polynomial r(x):', str(r_poly))

    # syndromes
    sz = coder._syndromes(r_poly)
    # sz is a Polynomial reversed; show coefficients
    print('\nSyndrome polynomial s(z):', str(sz))
    # list syndromes
    s_list = [sz.get_coefficient(i) for i in range(sz.degree(), -1, -1)]
    print('Syndrome coefficients (S1..):', [int(x) for x in s_list if x is not None])

    # Berlekamp-Massey
    sigma, omega = coder._berlekamp_massey(sz)
    print('\nError locator polynomial sigma(z):', str(sigma))
    print('Error evaluator polynomial omega(z):', str(omega))

    # Chien search
    X, j = coder._chien_search(sigma)
    print('\nChien search results:')
    print('Error locations X (field elements):', [int(x) for x in X])
    print('Error positions j (indices):', j)

    # Forney
    Y = coder._forney(omega, X)
    print('\nError magnitudes Y:', [int(y) for y in Y])

    # Build error polynomial and correct
    Elist = []
    for i in range(coder.n):
        if i in j:
            Elist.append(Y[j.index(i)])
        else:
            Elist.append(GF256int(0))
    E = Polynomial(reversed(Elist))
    print('Error polynomial E(x):', str(E))

    c_corrected = r_poly - E
    print('\nCorrected codeword polynomial c(x):', str(c_corrected))
    cw_bytes = "".join(chr(int(x)) for x in c_corrected.coefficients).rjust(coder.n, "\0")
    print('Corrected bytes repr:', repr(cw_bytes))
    print('Corrected bits:', bits_of_bytes(cw_bytes))

    decoded = ''.join(chr(int(x)) for x in c_corrected.coefficients[:-(coder.n - coder.k)])
    print('\nDecoded message:', repr(decoded))
    # print('Decoded:', repr(decoded.lstrip('\0')))
    return c_corrected, decoded

def plot_polynomial_evaluations(coder: RSCoder, poly: Polynomial, nsym: int):
    """Plot polynomial evaluations at field elements α^1..α^n and mark the
    evaluation points used for syndromes (α^1..α^nsym).
    """

    n = coder.n
    p = GF256int(3)
    xs = list(range(1, n + 1))
    ys = []
    for i in xs:
        val = poly.evaluate(p ** i)
        ys.append(int(val))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, marker='o', linestyle='-', label='poly(α^i)')
    # mark syndrome evaluation points
    synd_x = list(range(1, nsym + 1))
    synd_y = [ys[i - 1] for i in synd_x]
    ax.scatter(synd_x, synd_y, color='red', s=80, label='syndrome points')
    for x, y in zip(synd_x, synd_y):
        ax.annotate(str(y), (x, y), textcoords='offset points', xytext=(0, 6), ha='center')

    ax.set_xlabel('i (evaluation at α^i)')
    ax.set_ylabel('value (as integer)')
    ax.set_title('Polynomial evaluations over GF(2^8) (values shown as ints)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def demo():
    coder2 = RSCoder(7, 4)
    c_poly2, c_bytes2 = visualize_encode(coder2, "0001")
    r2 = "1" + c_bytes2[1:]
    visualize_decode(coder2, r2)

    # plot_polynomial_evaluations(coder2, c_poly2, coder2.n - coder2.k)

    # coder = RSCoder(7, 4)
    # c_poly, c_bytes = visualize_encode(coder, "0001")
    # r2 = c_bytes
    # visualize_decode(coder, r2)


if __name__ == '__main__':
    demo()


