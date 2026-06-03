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

Crack width is estimated in two ways:

1. Area/length average width:

```text
mean_width_px = mask_area_px / skeleton_length_px
mean_width_mm = mean_width_px * GSD
```

2. Skeleton distance-transform width:

```text
local_width_px = 2 * distance_to_mask_boundary_at_skeleton_point
```

The CSV reports mean, median, 95th percentile, and max values for the
distance-transform width. Width is less stable than length because it depends
strongly on segmentation boundary quality and threshold.

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

At this GSD, a `4000x3000` image approximately covers:

```text
width  = 4000 * 1.075 mm = 4.300 m
height = 3000 * 1.075 mm = 3.225 m
area   = 13.868 m^2
```

## Usage

Measure from probability maps:

```bash
python scripts/measure_crack_length.py \
  --input outputs/rerun_16imgs_b48_tile400_overlap96_thr04/probs \
  --output-csv outputs/measurements/b48_thr04_lengths.csv \
  --threshold 0.4 \
  --gsd-mm-per-px 1.075 \
  --skeleton-dir outputs/measurements/b48_thr04_skeletons \
  --base-image-dir outputs/rerun_16imgs_b48_tile400_overlap96_thr04/overlays \
  --annotated-dir outputs/measurements/b48_thr04_length_overlays
```

Optional small-component filtering:

```bash
python scripts/measure_crack_length.py \
  --input outputs/rerun_16imgs_b48_tile400_overlap96_thr04/probs \
  --output-csv outputs/measurements/b48_thr04_lengths_min50.csv \
  --threshold 0.4 \
  --gsd-mm-per-px 1.075 \
  --min-area-px 50 \
  --skeleton-dir outputs/measurements/b48_thr04_skeletons_min50 \
  --base-image-dir outputs/rerun_16imgs_b48_tile400_overlap96_thr04/overlays \
  --annotated-dir outputs/measurements/b48_thr04_length_overlays_min50 \
  --min-label-length-mm 50
```

Length-annotated overlays contain:

- cyan skeleton centerline
- total crack length and mean crack width in the upper-left corner
- component bounding boxes
- labels for the longest connected skeleton components, including component
  length and median width

Use `--max-labels` to limit label density and `--min-label-length-mm` to hide
short noisy fragments. Component labels follow this style:

```text
#1 L=35.2cm W~2.8mm
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
