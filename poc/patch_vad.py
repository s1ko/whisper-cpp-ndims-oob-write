import struct
import sys

src = sys.argv[1]
dst = sys.argv[2]
new_nlayers = int(sys.argv[3], 0) if len(sys.argv) > 3 else 1

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

magic = ru32()
assert magic == 0x67676d6c, f'bad magic {magic:#x}'

str_len = ri32()
model_type = bytes(b[pos:pos + str_len]).decode('latin1')
pos += str_len
print(f'model_type={model_type!r} str_len={str_len}')

major, minor, patch = ri32(), ri32(), ri32()
n_window = ri32()
n_context = ri32()
print(f'version={major}.{minor}.{patch} n_window={n_window} n_context={n_context}')

nlayers_off = pos
orig_nlayers = ri32()
print(f'n_encoder_layers @ {nlayers_off} = {orig_nlayers}')
for i in range(orig_nlayers):
    inc, outc, ks = ri32(), ri32(), ri32()
    print(f'  layer[{i}] in={inc} out={outc} kernel={ks}')

struct.pack_into('<i', b, nlayers_off, new_nlayers)
with open(dst, 'wb') as f:
    f.write(b)
print(f'patched n_encoder_layers {orig_nlayers} -> {new_nlayers} at offset {nlayers_off}; wrote {dst} ({len(b)} bytes)')
