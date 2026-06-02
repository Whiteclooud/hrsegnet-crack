# Crack Length Measurement

This project measures crack length in two layers:

1. Image layer: mask to skeleton centerline to pixel length.
2. Physical layer: pixel length to real length using a fixed GSD first.

## Current Measurement Definition

The current implementation measures the curve length of the crack centerline.
It skeletonizes the binary crack mask and sums 8-neighbor skeleton edges:

- horizontal/vertical edge: `1 px`
- diagonal edge: `sqrt(2) px`

This is different from endpoint distance. Curved cracks are measured along the
centerline curve.

## Fixed GSD

For the current DJI M3TD ZoomCamera images:

- image size: `4000x3000`
- relative altitude: about `20.013 m`
- focal length: `29.9 mm`
- 35mm equivalent focal length: `161 mm`
- gimbal pitch: about `-89.90`

Using the 35mm diagonal-equivalent focal length gives an approximate GSD:

```text
image diagonal = 5000 px
35mm diagonal = 43.27 mm
GSD = H * 43.27 / (f35 * image_diagonal)
    = 20.013 * 43.27 / (161 * 5000)
    = 0.001075 m/px
    = 1.075 mm/px
```

This is only a first-pass approximation. It assumes the photographed surface is
approximately planar, near perpendicular to the camera optical axis, and that
`RelativeAltitude` is close to the camera-to-surface distance.

## Usage

Measure from probability maps:

```bash
python scripts/measure_crack_length.py \
  --input outputs/rerun_16imgs_b48_tile400_overlap96_thr04/probs \
  --output-csv outputs/measurements/b48_thr04_lengths.csv \
  --threshold 0.4 \
  --gsd-mm-per-px 1.075 \
  --skeleton-dir outputs/measurements/b48_thr04_skeletons
```

Optional small-component filtering:

```bash
python scripts/measure_crack_length.py \
  --input outputs/rerun_16imgs_b48_tile400_overlap96_thr04/probs \
  --output-csv outputs/measurements/b48_thr04_lengths_min50.csv \
  --threshold 0.4 \
  --gsd-mm-per-px 1.075 \
  --min-area-px 50 \
  --skeleton-dir outputs/measurements/b48_thr04_skeletons_min50
```

## Broken And Curved Cracks

The first version reports connected skeleton components separately through
component counts and max component length. It does not merge gaps by default.

For long cracks broken into many small pieces, future logic should support
gap-aware merging using:

- endpoint distance
- endpoint direction similarity
- local alignment or curve continuation
- maximum allowed gap length

Until that is implemented, use the total skeleton length as "all detected crack
length" and component counts as a warning that the crack is fragmented.
