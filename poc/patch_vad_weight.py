import struct
import sys

src = sys.argv[1]
dst = sys.argv[2]
new_ndims = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x40000000

with open(src, 'rb') as f:
    b = bytearray(f.read())

pos = 0

def ri32():
    global pos
    v = struct.unpack_from('<i', b, pos)[0]
    pos += 4
    return v

def ru32():
    global pos
    v = struct.unpack_from('<I', b, pos)[0]
    pos += 4
    return v

assert ru32() == 0x67676d6c, 'bad magic'
str_len = ri32()
pos += str_len
ri32(); ri32(); ri32()        # major, minor, patch
ri32(); ri32()                # n_window, n_context
n_layers = ri32()
for _ in range(n_layers):
    ri32(); ri32(); ri32()    # per-layer triple
ri32(); ri32(); ri32(); ri32()  # lstm_input, lstm_hidden, final_conv_in, final_conv_out

first_weight_off = pos
orig = struct.unpack_from('<i', b, pos)[0]
length = struct.unpack_from('<i', b, pos + 4)[0]
ttype = struct.unpack_from('<i', b, pos + 8)[0]
print(f'first weight record @ {first_weight_off}: n_dims={orig} length={length} ttype={ttype} (n_encoder_layers={n_layers} kept)')

struct.pack_into('<i', b, first_weight_off, new_ndims)
with open(dst, 'wb') as f:
    f.write(b)
print(f'patched first weight n_dims {orig} -> {new_ndims:#x} at offset {first_weight_off}; wrote {dst}')
