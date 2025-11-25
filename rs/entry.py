import rs
from polynomial import Polynomial
from ff import GF256int


def bitstring_from_bytes(s: str) -> str:
	"""Return continuous bit string and grouped bytes for display."""
	ints = [ord(c) for c in s]
	grouped = ' '.join(f"{b:08b}" for b in ints)
	continuous = ''.join(f"{b:08b}" for b in ints)
	return continuous, grouped


def poly_from_bytes(s: str) -> Polynomial:
	# build polynomial from bytes (chars) using GF256int coefficients
	return Polynomial(GF256int(ord(x)) for x in s)


coder = rs.RSCoder(20,13)
c_poly = coder.encode("Hello, world!", poly=True)
c = "".join(chr(int(x)) for x in c_poly.coefficients).rjust(coder.n, "\0")
print('Encoded codeword (as bytes repr):', repr(c))
continuous, grouped = bitstring_from_bytes(c)
print('Encoded codeword bits (grouped bytes):', grouped)
print('Encoded codeword bits (continuous):', continuous)
print('Encoded polynomial c(x):', str(c_poly))

r = "\0"*3 + c[3:]
print('\nReceived (with first 3 bytes zeroed):', repr(r))
rc_poly = poly_from_bytes(r)
print('Received polynomial r(x):', str(rc_poly))

result = coder.decode(r)
print('Decoded result:', repr(result))


coder = rs.RSCoder(7,4)
c_poly2 = coder.encode("0001", poly=True)
c2 = "".join(chr(int(x)) for x in c_poly2.coefficients).rjust(coder.n, "\0")
print('\nSecond example:')
print('Encoded (bytes):', repr(c2))
cont2, grp2 = bitstring_from_bytes(c2)
print('Bits grouped:', grp2)
print('Polynomial:', str(c_poly2))

r2 = "1" + c2[1:]
print('Received (with first symbol flipped):', repr(r2))
print('Received poly:', str(poly_from_bytes(r2)))
result2 = coder.decode(r2)
print('Decoded result:', repr(result2))