# whisper.cpp: stack out-of-bounds write from an unvalidated tensor `n_dims` in model loading

`whisper_model_load()` reads a 32-bit dimension count, `n_dims`, from every
tensor record of a legacy GGML model file and uses it as the loop bound for
writes into a fixed `int32_t ne[4]` array on the stack. Nothing checks that
`n_dims <= 4`, so a record with a larger value makes `read_safe(loader,
ne[i])` write attacker-supplied 4-byte values past the end of the array. The
file is parsed when the model is loaded, before any inference, so passing a
crafted `ggml-*.bin` to `whisper_init_from_file()` (or to `whisper-cli -m`)
crashes the process. The identical loop in `whisper_vad_init_with_params()`
is reached through `--vad-model`.

> ## Fixed in whisper.cpp v1.9.4
>
> Fixed upstream by
> [`df1547b6`](https://github.com/ggml-org/whisper.cpp/commit/df1547b6fe362556e48fd21f386c6eee85b971fe)
> (PR [#3957](https://github.com/ggml-org/whisper.cpp/pull/3957), merged
> 2026-08-07), which rejects `n_dims < 0 || n_dims > 4` in the whisper, VAD
> and parakeet loaders. The first release containing it is the `v1.9.3`
> prerelease (2026-08-20); the first stable release is **v1.9.4**
> (2026-09-11). The latest release, `v1.9.5`, was re-tested and rejects all
> models in this repository. This was a publicly reported issue before this
> report was filed; see [Prior coverage](#prior-coverage). The fix does not
> credit this report.

| | |
|---|---|
| CVE | none assigned for `ggml-org/whisper.cpp`. A vendored copy of the same loader in Praat was assigned CVE-2026-110698; see Prior coverage. |
| CWE | CWE-787 (Out-of-bounds Write), CWE-121 (Stack-based Buffer Overflow) |
| Project | [`ggml-org/whisper.cpp`](https://github.com/ggml-org/whisper.cpp) |
| Affected | up to and including `v1.9.2` (2026-08-04). Tested: `v1.9.2` built with AddressSanitizer. Inspected: the same unbounded loop, with no `n_dims` check, is present in `v1.5.0`, `v1.7.0`, `v1.8.0` and `v1.9.1`. Also repository HEAD `fc674574ca27cac59a15e5b22a09b9d9ad62aafe` (2026-04-20) at the time of the report. |
| Fixed in | `v1.9.3` (prerelease), `v1.9.4` (first stable) |
| CVSS v3.1 | `AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:H` = **6.1 Medium**. `AV:N` = 7.1 where a service loads uploaded models. |
| Reported by | s1ko ([github.com/s1ko](https://github.com/s1ko), `s1ko@riseup.net`) |

## Scope, stated plainly

What is demonstrated is memory corruption of the stack by file-controlled
data and a reliable crash of the loading process. The written values and the
number of 4-byte writes are chosen by the file. No control-flow hijack was
built, and whether one is possible depends on the compiler's stack protector,
the platform and the surrounding frame layout; this advisory does not claim
code execution. Integrity impact is scored `L` for that reason.

The AddressSanitizer build used here does not print a
`stack-buffer-overflow` report for this file. The out-of-bounds writes are
performed by `std::ifstream::read` inside the uninstrumented C++ runtime, which
the sanitizer does not check. The writes clobber stack memory beyond `ne[]`,
most likely including adjacent locals such as the loop counter, and the
process faults shortly afterwards at an unmapped stack address (SEGV at
`whisper.cpp:1886`). The
stack-write form of the report, `WRITE of size 4`, was captured by other
reporters with different builds; see Prior coverage.

## Threat model

Whisper model files are large binary blobs that users download from model
hubs, mirrors and third parties and treat as data. Applications that embed
whisper.cpp (transcription front ends, services that accept a user-supplied
model, model conversion and CI tooling) call `whisper_init_from_file()` or
`whisper_init_from_file_with_params()` on them. The attacker supplies only the
file; no in-memory object is needed, and the user's action is loading the
model (`UI:R`).

Reachability: `whisper_init_from_file_with_params` →
`whisper_init_from_file_with_params_no_state` (opens the path with
`std::ifstream` and builds a loader whose `read` calls `fin->read`) →
`whisper_init_with_params_no_state` → `whisper_model_load`. The header
fields before the weight loop (magic, hyperparameters, mel filters, vocabulary)
are read permissively, so a crafted file reaches its first tensor record.

## Execution

### Root cause

`src/whisper.cpp` at `v1.9.2` (`306c88f4d1286aec1bf96e544632897886af5501`),
`whisper_model_load`, lines 1869-1887:

```cpp
        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            read_safe(loader, n_dims);
            read_safe(loader, length);
            read_safe(loader, ttype);

            if (loader->eof(loader->context)) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[4] = { 1, 1, 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                read_safe(loader, ne[i]);
                nelements *= ne[i];
            }
```

with `read_safe` at line 962:

```cpp
template<typename T>
static void read_safe(whisper_model_loader * loader, T & dest) {
    loader->read(loader->context, &dest, sizeof(T));
    BYTESWAP_VALUE(dest);
}
```

`n_dims` (line 1874) comes straight from the file and bounds the loop on line
1884. For `i >= 4`, `&ne[i]` lies beyond the 16-byte array and `read_safe`
stores four file-controlled bytes there on every iteration. The shape and size
validation that would reject a bad record (lines 1894-1920, `unknown tensor`,
`wrong size`, `wrong shape`) all run after the loop.

The VAD loader, `whisper_vad_init_with_params`, has the same code at lines
5011-5029 (`int32_t ne[4]` on line 5025, the loop on lines 5026-5029). It is
reached from the ordinary transcription path whenever VAD is enabled with a
user-supplied VAD model (`whisper_full` → `whisper_vad` →
`whisper_vad_init_from_file_with_params`).

### Reproduction

[`poc/`](poc/) contains the three generators and
[`poc/README.md`](poc/README.md) the build and run commands. The models are
derived from fixtures in the whisper.cpp tree rather than stored here.

```sh
git clone https://github.com/ggml-org/whisper.cpp.git && cd whisper.cpp
git checkout v1.9.2
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DWHISPER_SANITIZE_ADDRESS=ON \
      -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
ninja -C build whisper-cli

python3 -I ../poc/patch_ndims.py models/for-tests-ggml-tiny.en.bin evil-whisper-ndims.bin 8
ASAN_OPTIONS=detect_leaks=0 ./build/bin/whisper-cli -m evil-whisper-ndims.bin -f samples/jfk.wav
```

`patch_ndims.py` parses the fixture's header (magic, 11 hyperparameters, mel
filters, vocabulary) to find the first tensor record, and appends one
malicious record: `n_dims`, `length = 4`, `ttype = 0`, then 65536 bytes of
`0x41`. The unmodified fixture is the control: it loads and runs normally.

Observed on `v1.9.2`, 2026-10-28 (paths shortened, addresses elided):

```
==ERROR: AddressSanitizer: SEGV on unknown address 0x7ffe03f9d9e4 (pc ... T0)
    #0 in whisper_model_load src/whisper.cpp:1886
    #1 in whisper_init_with_params_no_state src/whisper.cpp:3730
    #2 in whisper_init_from_file_with_params_no_state src/whisper.cpp:3659
    #3 in whisper_init_from_file_with_params src/whisper.cpp:3750
SUMMARY: AddressSanitizer: SEGV src/whisper.cpp:1886 in whisper_model_load
```

Line 1886 is `nelements *= ne[i];` inside the unbounded loop. The same crash
occurs with `n_dims` values of 8, 16, 64 and `0x40000000`. A value of 5 or 6
does not fault in this build: the extra 4-byte writes land in stack padding,
and the loader goes on to reject the record as `unknown tensor 'AAAA'` (the
four `0x41` bytes that follow are read as the tensor name, which is simply the
next bytes of the file). The overflow happens in that case too, but is not
observable from the exit status.

VAD loader (`evil-vad-ndims.bin`, first weight record's `n_dims` set to
`0x40000000`):

```
==ERROR: AddressSanitizer: SEGV on unknown address 0x7ffc7e4d7278 (pc ... T0)
    #0 in whisper_vad_init_with_params src/whisper.cpp:5028
    #1 in whisper_vad_init_from_file_with_params src/whisper.cpp:4766
    #2 in whisper_vad src/whisper.cpp:6667
    #3 in whisper_full src/whisper.cpp:7781
```

On `v1.9.5` the same files are rejected and the process exits non-zero:

```
whisper_model_load: invalid n_dims 8 in model file (expected 0 <= n_dims <= 4)
whisper_vad_init_with_params: invalid n_dims 1073741824 in model file (expected 0 <= n_dims <= 4)
```

### Related: VAD `n_encoder_layers`

`whisper_vad_init_with_params` also read `n_encoder_layers` from the VAD model
header and used it unvalidated to size three arrays, while the graph it builds
indexes them at `[0..3]` and `n_tensors` is derived from it. `patch_vad.py`
with a value of 1 makes `v1.9.2` abort (`ggml_new_object: not enough space in
the context's memory pool`, exit status 134); a negative value throws
`std::bad_array_new_length` (re-run on `v1.9.2`, exit status 134). The original
observation of a multi-gigabyte allocation for a very large value was not re-run.
This is a denial of service. It was fixed separately by
[`a44e0784`](https://github.com/ggml-org/whisper.cpp/commit/a44e07845931421bb6f3447ce0010ed9dc76a118)
(PR #4064, merged 2026-09-22, in `v1.9.5`), which requires the value to be 4.

## Detection

**Static, on the model file.** A GGML whisper model is a sequence of tensor
records after the header: three `int32` fields (`n_dims`, `length`, `ttype`),
`n_dims` `int32` extents, `length` bytes of name, then the tensor data. A
legitimate record has `0 <= n_dims <= 4` (the silero VAD model contains a
genuine rank-0 tensor, so 0 is valid). Any record outside that range is
malformed. Walking the records needs the header layout that
`poc/patch_ndims.py` implements; a record with `n_dims > 4` is the indicator.
A byte-pattern YARA rule is not offered, because the position of the first
record depends on the vocabulary and mel-filter sizes.

**Static, on source** (for forks and vendored copies):

```sh
grep -rn 'read_safe(loader, ne\[i\])' --include='*.cpp' .
```

and confirm each hit is preceded by a `n_dims < 0 || n_dims > 4` check.

**Runtime.** Process termination by SIGSEGV or SIGABRT (stack protector) with
`whisper_model_load` or `whisper_vad_init_with_params` in the backtrace,
immediately after a model file was opened, before any audio is processed. On
a patched build the signal is a log line,
`invalid n_dims <N> in model file (expected 0 <= n_dims <= 4)`, and a failed
init; alert on that string, it is only produced by a malformed model.

```yaml
title: whisper.cpp rejects or crashes on a malformed model file
id: 5c9d7a3e-1f0b-4e4a-8c53-9a2d4b6e7f10
status: experimental
description: |
  Detects abnormal termination by SIGSEGV of a process that links whisper.cpp
  while it is loading a model. On versions up to v1.9.2 a crafted tensor
  n_dims overflows a stack array in whisper_model_load. Crash telemetry is the
  external signal; correlate with the model file that was just supplied.
references:
  - https://github.com/ggml-org/whisper.cpp/pull/3957
logsource:
  product: linux
  category: process_termination
detection:
  selection:
    Image|endswith:
      - '/whisper-cli'
      - '/whisper-server'
    ExitSignal: 11
  condition: selection
falsepositives:
  - Other crashes of the same programs
level: low
```

## Mitigation

- **Upgrade to whisper.cpp v1.9.4 or later** (v1.9.5 is the current release).
  Bindings and downstream packages that vendor `src/whisper.cpp` need the same
  change, `df1547b6`, applied to their copy.
- Apply the guard yourself if you cannot upgrade, in both loaders and in
  `parakeet_model_load` if you carry it:

  ```cpp
  if (n_dims < 0 || n_dims > 4) {
      WHISPER_LOG_ERROR("%s: invalid n_dims %d in model file\n", __func__, n_dims);
      return false;
  }
  ```

  `n_dims == 0` is valid and must not be rejected: the shipped silero VAD
  model contains a rank-0 tensor.
- **Do not load models from sources you do not control**, or verify a hash
  published by the model's author before loading.
- **Contain the load.** Run transcription of third-party models in an
  unprivileged, sandboxed process so that a crash or memory corruption costs
  one job.
- Build with a stack protector and `_FORTIFY_SOURCE` (distribution defaults);
  this turns most exploitation attempts into aborts. It does not remove the
  bug.

Mapping: CWE-787 and CWE-121 mitigations (bounds validation before write,
stack canaries); NIST SP 800-53r5 `SI-10`, `SI-16`; NIST SP 800-218 `PW.5`,
`PW.8`; MITRE ATT&CK mitigation M1048 (Application Isolation and Sandboxing).

## Timeline

| Date | Event |
|---|---|
| 2026-04-20 | Repository HEAD `fc674574`, the revision this report was written against |
| 2026-05-01 | [whisper.cpp#3787](https://github.com/ggml-org/whisper.cpp/issues/3787) opened by another reporter, public, same defect with an AddressSanitizer trace on `fc674574` |
| 2026-05-30 | Reported via huntr.com (report 3cab0f26-9ef0-45f5-abfb-fca19f0b0b92). No triage, validation or response from the platform as of 2026-10-28. |
| 2026-07-18 | [#3944](https://github.com/ggml-org/whisper.cpp/issues/3944) opened by another reporter |
| 2026-07-27 | PR #3957 opened |
| 2026-08-04 | `v1.9.2` released; still affected |
| 2026-08-07 | PR #3957 merged (`df1547b6`) |
| 2026-08-20 | `v1.9.3` prerelease, first release with the fix |
| 2026-09-06 | #3787 closed by the stale-issue bot without a linked fix |
| 2026-09-11 | `v1.9.4`, first stable release with the fix |
| 2026-09-22 | PR #4064 merged (`a44e0784`): VAD `n_encoder_layers` check, in `v1.9.5` |
| 2026-10-06 | `v1.9.5` released; re-tested, all models in this repository rejected |
| 2026-10-28 | Published |

## Prior coverage

This is not a first discovery. The defect was reported publicly 29 days
before the Huntr submission:

- [**ggml-org/whisper.cpp#3787**](https://github.com/ggml-org/whisper.cpp/issues/3787),
  opened 2026-05-01 by `qi-scape`: "Stack buffer overflow WRITE in
  whisper_model_load via crafted GGML model (n_dims)". It describes the same
  `n_dims >= 5` write past `ne[4]` at `src/whisper.cpp:1884`, tested on
  `fc674574` (the same revision as this report), with an AddressSanitizer
  `stack-buffer-overflow ... WRITE of size 4` trace and a PoC file. It also
  notes the same pattern in the VAD loader. The issue was closed by the
  stale-issue bot on 2026-09-06; no fix was linked to it.
- [**#3944**](https://github.com/ggml-org/whisper.cpp/issues/3944), opened
  2026-07-18 by `professor-moody`, reports the same `n_dims` write in both
  loaders with a patch proposal; a comment of 2026-07-31 corrects the lower
  bound (rank 0 is valid). That issue is still open.
- PR [#3957](https://github.com/ggml-org/whisper.cpp/pull/3957) by
  `shafiuzzaman-md` (opened 2026-07-27, merged 2026-08-07) is the fix. The
  independent PR #3972 (2026-08-05), which proposed rejecting `n_dims` outside 1..4, was closed without merging.
- CVE-2026-110698 (VulDB, published 2026-10-14) covers the same
  `whisper_model_load` `n_dims` write in the copy of whisper.cpp vendored in
  Praat up to 7.0.02, tracked in praat/praat.github.io#3348; Praat applied its
  own quick fix.
- No GitHub Security Advisory exists for `ggml-org/whisper.cpp` for this
  defect.

The Huntr submission of 2026-05-30 is an independent report of the same bug:
it was written against the same revision, but it was not public, and it is
later than #3787. The VAD-loader and `n_encoder_layers` observations in this
repository were validated by running the shipped test fixtures with patched
headers, as in `poc/`.

## References

- Fix — https://github.com/ggml-org/whisper.cpp/pull/3957 (`df1547b6fe362556e48fd21f386c6eee85b971fe`)
- Earlier public reports — https://github.com/ggml-org/whisper.cpp/issues/3787, https://github.com/ggml-org/whisper.cpp/issues/3944
- Related VAD fix — https://github.com/ggml-org/whisper.cpp/pull/4064
- Vendored-copy CVE — CVE-2026-110698 (Praat), https://github.com/praat/praat.github.io/issues/3348
- Releases — https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.4, https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.5
- PoC models — https://huggingface.co/s1ko/whisper-cpp-ggml-ndims-oob-poc

## License

MIT, see [LICENSE](LICENSE).
