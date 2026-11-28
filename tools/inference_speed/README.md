# Inference speed tools (#33)

- `prof_tile.py`: profile the reference predictor on a few tiles (CUDA time per module, top operations).
- `bench_engines.py`: A/B of the engines on one cached trial-path. It reports the time, the GPU utilisation and the differences from the reference in counts.
- `macs.py`: convolution multiply-accumulates per output voxel, for the tile and streaming geometries and the low-rate designs.
- `time_load.py`: load and stats with and without `--fast_read` / `--valid_stats`, as interleaved pairs.
- `decimate.py`: the decimation test's baselines (block means at 40, 20 and 10 Hz, interpolated back) and the baseline spectrum.

Results and the run of 2026-11-27: `docs/plan.md`, "Inference speed".
