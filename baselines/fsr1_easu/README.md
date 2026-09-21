# FSR 1 (EASU + RCAS) -- CPU reference port

A faithful CPU port of AMD FidelityFX Super Resolution 1.0's EASU (edge-adaptive
spatial upsampling) and RCAS (robust contrast-adaptive sharpening), translated
directly from the official MIT-licensed `ffx_fsr1.h` (vendored at
`vendor/ffx_fsr1.h` and `vendor/ffx_a.h`; see `vendor/LICENSE.txt` and the
repo's `THIRD_PARTY.md`).

## Framing: this is a quality baseline, not a performance baseline

**The real FSR 1 runs as a GPU compute shader.** This port exists purely so
`evaluate.py` can compare FrameForge's output quality against FSR 1's, on
CPU, offline. Never report this binary's CPU latency as "FSR 1 performance"
anywhere in this project's results -- see CLAUDE.md 6.1.

## Usage

```
fsr1_cli --in in.png --out out.png --width W --height H [--sharpness S]
```

`--sharpness` is optional; when given, RCAS runs after EASU (`0.0` = maximum
sharpness, larger values = fewer stops of sharpening, matching the original's
convention). Without it, only EASU runs.

`evaluate.py` calls this binary via subprocess per CLAUDE.md 6.1's "expose it
via a CLI ... or a subprocess for evaluation" option.

## Known deviations from the original GPU shader

1. **Exact reciprocal/rsqrt instead of the GPU's fast approximations.** The
   original uses `APrxLoRcpF1` / `APrxLoRsqF1` / `APrxMedRcpF1` -- bit-hack
   approximations chosen for shader throughput. This CPU port uses exact
   `1/x` and `1/sqrt(x)` throughout, guarded against division by zero (see
   `safe_rcp` in `src/fsr1.cpp`). This is strictly more accurate and fully
   deterministic; it is the one intentional algorithmic substitution.
2. **No packed-FP16 path.** The GPU shader has an optional 16-bit packed
   variant purely for GPU throughput; meaningless on CPU, not ported.
3. **Clamped-edge sampling at image borders**, matching a GPU texture
   sampler's clamp addressing mode.

## Visual validation

Manually checked against a real project fixture frame
(`tests/fixtures/sequences/synthetic_pan_00/lr/frame_0001.exr`, exported to
PNG): EASU's 2x upscale preserves the source's smooth gradient structure
with no artifacts, NaNs, or corrupted output; RCAS visibly increases local
contrast at color transitions when run at `--sharpness 0.0`. The synthetic
fixture is a smooth analytic pattern with no hard edges, so it does not
exercise EASU's edge-adaptive anisotropic behavior strongly -- that is
better judged once real rendered content (with actual geometric edges) is
available (see `docs/DECISIONS.md` D0/D5).

## Tests

`tests/fsr1_golden_test.cpp` (GoogleTest): determinism (bit-identical
output across repeated runs on a fixed synthetic input), shape/range
sanity, a pixel-value regression check against values captured from a real
run of this code, and an RCAS sharpness-monotonicity check.
