"""Maps GTSRB class IDs to semantically-equivalent classes in the BelgiumTSC dataset
(https://www.kaggle.com/datasets/abhi8923shriv/belgium-ts, CC0), a second out-of-
distribution generalization check alongside Mapillary+DFG (src/data/mapillary_mapping.py).

BelgiumTSC ships no official class-ID-to-meaning file (unlike Mapillary+DFG's
classes.json). Classes were identified by visually inspecting sample images from each of
the 62 class folders and comparing against GTSRB_CLASS_NAMES, then **validated by
checking what the GTSRB-trained MobileNetV2 predicts for ~20 images per candidate class**
-- a mismatch between the visual guess and the model's consistent prediction was a
reliable signal of a wrong label, not just "hard to generalize."

That validation caught real errors from single-thumbnail visual identification (all
between visually-similar triangular/circular pictograms at low resolution):
- Class 00022 looked like "stop" at thumbnail size; it's actually "no entry" (plain
  red circle, white bar) -- the model predicted "no entry" for 20/20 samples.
- Class 00007 looked like "road work" (a humanoid figure); it's actually "children
  crossing" (two child figures) -- confirmed by both closer visual inspection and the
  model's predictions clustering on "children crossing".
- Class 00017 looked like "pedestrians" (a person-shaped icon); side-by-side comparison
  against real GTSRB class 11 and 27 reference images confirmed it's actually
  "right-of-way at next intersection" (GTSRB 11's icon, not GTSRB 27's walking figure).
- Class 00020 looked like "no passing" (two arrow-like shapes); closer inspection showed
  a directional priority-arrows pictogram with no GTSRB equivalent, not two vehicles.
The genuine "road work" (00010, person digging with a dirt pile) and "stop" (00021,
literal octagonal "STOP" text) signs were found elsewhere in the 62 classes.

Two GTSRB classes have no confident BelgiumTSC match at all: "no passing" (9, no
two-vehicle-overtaking icon found) and "pedestrians" (27, no single-walking-figure icon
found) -- not necessarily absent from Belgium as a country, just not present in this
particular 62-class reduced subset.

Other exclusions (never had a viable candidate to begin with), same policy as the
Mapillary mapping:
- Speed-limit classes: BelgiumTSC's speed signs are diamond-shaped advisory signs, not
  GTSRB's circular regulatory ones -- a different sign category, not a rendering
  difference.
- Direction-specific single curves: GTSRB separates "dangerous curve left" (19) from
  "...right" (20), and curve direction isn't reliably distinguishable from these
  low-resolution dashcam-style crops -- an incorrect direction guess would be worse than
  no match. The *double* curve (00006) is included since GTSRB 21 has no directional
  sub-variant.
- Symmetric "road narrows" icons: GTSRB 24 is specifically "narrows on the right"
  (asymmetric icon); BelgiumTSC's candidates narrow from both sides.
- Farm-animal crossing (cow icon): a different Vienna Convention pictogram from GTSRB
  31's wild-animal (deer) icon, not a rendering variant of the same sign.
- Signs with no GTSRB equivalent at all (parking, no-stopping, height/weight limits
  beyond the one weight-limit match below, level crossings, one-way streets, mandatory
  bicycle/pedestrian paths, etc.).

Two classes (double curve, road work, turn left ahead) validated as visually correct but
still generalize poorly (see reports/belgium_generalization_results.json's per-class F1)
-- kept in the mapping since the label itself is right; the poor transfer is a genuine
finding, not a mapping artifact. See ROBUSTNESS_REPORT.md's BelgiumTSC section.
"""

from src.data.mapillary_mapping import GTSRB_CLASS_NAMES  # re-exported for convenience

# GTSRB class ID -> BelgiumTSC class ID (zero-padded folder name, e.g. "00022")
GTSRB_TO_BELGIUM = {
    11: "00017",  # right-of-way at next intersection
    12: "00061",  # priority road
    13: "00019",  # yield
    14: "00021",  # stop
    15: "00028",  # no vehicles
    16: "00025",  # no vehicles >3.5t
    17: "00022",  # no entry
    18: "00013",  # general caution
    21: "00006",  # double curve
    22: "00000",  # bumpy road
    23: "00002",  # slippery road
    25: "00010",  # road work
    26: "00011",  # traffic signals
    28: "00007",  # children crossing
    29: "00008",  # bicycles crossing
    34: "00035",  # turn left ahead
    35: "00034",  # ahead only
    36: "00036",  # go straight or right
    40: "00037",  # roundabout mandatory
}
