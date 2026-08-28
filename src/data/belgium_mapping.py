"""Maps GTSRB class IDs to semantically-equivalent classes in the BelgiumTSC dataset
(https://www.kaggle.com/datasets/abhi8923shriv/belgium-ts, CC0), a second out-of-
distribution generalization check alongside Mapillary+DFG (src/data/mapillary_mapping.py).

BelgiumTSC ships no official class-ID-to-meaning file (unlike Mapillary+DFG's
classes.json) -- its 62 classes were identified by visually inspecting sample images from
each class folder and comparing against GTSRB_CLASS_NAMES (see mapillary_mapping.py).
Only classes with a confident, unambiguous visual match are included. Notably excluded,
same policy as the Mapillary mapping:
- Speed-limit classes: BelgiumTSC's speed signs are diamond-shaped advisory signs, not
  GTSRB's circular regulatory ones -- different sign category, not just a rendering
  difference.
- Direction-specific single curves (BelgiumTSC 00003/00004): GTSRB separates "dangerous
  curve left" (19) from "...right" (20), and the curve direction isn't reliably
  distinguishable from these particular low-resolution dashcam-style crops -- an
  incorrect direction guess would be worse than no match at all. The *double* curve
  (00006) is included since GTSRB's class 21 has no directional sub-variant.
- Symmetric "road narrows" (00014, 00016): GTSRB 24 is specifically "narrows on the
  right" (asymmetric icon); these BelgiumTSC icons narrow from both sides.
- Farm-animal crossing (00009, cow icon): a different Vienna Convention pictogram from
  GTSRB 31's wild-animal (deer) icon, not a rendering variant of the same sign.
- Signs with no GTSRB equivalent at all (parking, no-stopping, height/weight limits
  beyond the one weight-limit match below, level crossings, one-way streets, etc.).

GTSRB 17 ("no entry") has no match: this 62-class reduced subset doesn't appear to
include the plain white-bar-on-red-circle sign at all -- BelgiumTSC class 00023, which
looked like a plausible match at thumbnail resolution, turned out on closer inspection to
be "no bicycles" (a red circle with a bicycle pictogram), a different sign entirely.
"""

from src.data.mapillary_mapping import GTSRB_CLASS_NAMES  # re-exported for convenience

# GTSRB class ID -> BelgiumTSC class ID (zero-padded folder name, e.g. "00022")
GTSRB_TO_BELGIUM = {
    9: "00020",  # no passing
    12: "00061",  # priority road
    13: "00019",  # yield
    14: "00022",  # stop
    15: "00028",  # no vehicles
    16: "00025",  # no vehicles >3.5t
    18: "00013",  # general caution
    21: "00006",  # double curve
    22: "00000",  # bumpy road
    23: "00002",  # slippery road
    25: "00007",  # road work
    26: "00011",  # traffic signals
    27: "00017",  # pedestrians
    29: "00008",  # bicycles crossing
    34: "00035",  # turn left ahead
    35: "00034",  # ahead only
    36: "00036",  # go straight or right
    40: "00037",  # roundabout mandatory
}
