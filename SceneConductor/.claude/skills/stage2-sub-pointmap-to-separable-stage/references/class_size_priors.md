# Real-world size priors by object class

This table drives the `--world-scale auto` calculation. For each class we record a rough typical **longest-side dimension in meters**. The loader uses `target_m / predicted_scale` per classified object and takes the median to produce a single world-scale factor.

## Source conventions

- Values target **longest visible dimension**, not always tallest.
- When a class has wide variance (e.g., "table" can be 0.5–3 m), we pick a middle-ground dwelling-furniture value.
- Wall-mounted objects (window, curtain, picture_frame) use the window/frame **height**.

## Current table

```
sofa              1.9      armchair          0.9
chair             0.5      stool             0.45
coffee_table      1.0      table             1.5
desk              1.4      dining_table      1.8
pool_table        2.4
cushion           0.5      pillow            0.5
curtain           1.8      window            1.2
picture_frame     0.6      wall_art          0.6      mirror            0.8
bed               2.0      nightstand        0.5
dresser           1.3      wardrobe          1.8
lamp              0.4      floor_lamp        1.5      table_lamp        0.45
plant             0.6      vase              0.3
rug               2.0      carpet            2.5
tv                1.2      monitor           0.6      laptop            0.35
bookshelf         1.8      cabinet           1.0      shelf             1.0
refrigerator      1.8      stove             0.7      sink              0.6
toilet            0.7      bathtub           1.7      shower            1.8
door              2.1      clock             0.4
```

## Adding a new class

1. Add the entry to `CLASS_SIZE_PRIORS_M` in `src/convert.py`.
2. Pick the longest-side median size.
3. Use lowercase keys — the lookup is case-folded.

## Median, not mean

Median is robust to per-object scale noise. If the computed factor looks off (e.g., chairs render at 3 m or sofas at 30 cm), check the transcript log line `[world-scale] factors: [...]`. Override with `--world-scale <k>` and adjust the priors later.
