from rs import RSCoder
from visualize_rs import visualize_encode
from ff import GF256int

coder = RSCoder(7, 4)

# Encode message
c_poly, c_bytes = visualize_encode(coder, "0001")
print("c(x) =", c_poly)

r2 = "1" + c_bytes[1:]

# Evaluation points α^0 .. α^(n-1)
alpha = GF256int(3)
xs = [alpha ** i for i in range(coder.n)]
print("xs =", xs)

print("Adding a error at position 0")
r2 = "1" + c_bytes[1:]

# Evaluate codeword polynomial at these points
ys = [c_poly.evaluate(xi) for xi in xs]
print("ys =", [int(y) for y in ys])

# Convert to integers for BW decoder
r_eval = [int(y) for y in ys]
print("r_eval =", r_eval)

# Decode using BW, passing xs explicitly
recovered = coder.decode_berlekamp_welch(r_eval, xs=xs)
print("BW decoded:", recovered)
