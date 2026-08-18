"""Maps GTSRB class IDs to semantically-equivalent classes in the Mapillary+DFG dataset
(https://www.kaggle.com/datasets/nomihsa965/traffic-signs-dataset-mapillary-and-dfg),
used for the generalization check described in the project spec.

Only classes with a confident, unambiguous semantic match are included. Notably excluded:
- The 9 numeric speed-limit classes (GTSRB 0-8): Mapillary lumps every speed value into
  one generic "regulatory--maximum-speed-limit" class, so there's no way to recover which
  specific limit (20/30/.../120 km/h) a given crop shows without re-annotating by hand.
- Signs with no equivalent in this particular 76-class subset (e.g. priority road,
  right-of-way at next intersection, end-of-restriction signs) — this subset was
  curated for the Africa region and doesn't cover every European sign type.
- Ambiguous/multi-variant matches (e.g. GTSRB's undirected "double curve" vs.
  Mapillary's direction-specific "double-curve-first-left/right").

GTSRB_CLASS_NAMES documents what each GTSRB class ID actually depicts, since GTSRB itself
ships only numeric IDs (see src/data/dataset.py's class_names(), which just echoes the ID).
"""

GTSRB_CLASS_NAMES = {
    0: "speed limit 20", 1: "speed limit 30", 2: "speed limit 50", 3: "speed limit 60",
    4: "speed limit 70", 5: "speed limit 80", 6: "end speed limit 80", 7: "speed limit 100",
    8: "speed limit 120", 9: "no passing", 10: "no passing (>3.5t)",
    11: "right-of-way at next intersection", 12: "priority road", 13: "yield", 14: "stop",
    15: "no vehicles", 16: "no vehicles >3.5t", 17: "no entry", 18: "general caution",
    19: "dangerous curve left", 20: "dangerous curve right", 21: "double curve",
    22: "bumpy road", 23: "slippery road", 24: "road narrows right", 25: "road work",
    26: "traffic signals", 27: "pedestrians", 28: "children crossing",
    29: "bicycles crossing", 30: "beware ice/snow", 31: "wild animals crossing",
    32: "end speed/passing limits", 33: "turn right ahead", 34: "turn left ahead",
    35: "ahead only", 36: "go straight or right", 37: "go straight or left",
    38: "keep right", 39: "keep left", 40: "roundabout mandatory", 41: "end no passing",
    42: "end no passing (>3.5t)",
}

# GTSRB class ID -> Mapillary/DFG folder name (see data/mapillary/classes.json)
GTSRB_TO_MAPILLARY = {
    9: "regulatory--no-overtaking",
    13: "regulatory--yield",
    14: "regulatory--stop",
    17: "regulatory--no-entry",
    19: "warning--curve-left",
    20: "warning--curve-right",
    22: "warning--road-bump",
    23: "warning--slippery-road-surface",
    24: "warning--road-narrows-right",
    25: "warning--roadworks",
    26: "warning--traffic-signals",
    27: "warning--pedestrians-crossing",
    28: "warning--children",
    29: "warning--bicycles-crossing",
    31: "warning--wild-animals",
    33: "regulatory--turn-right",
    34: "regulatory--turn-left",
    35: "regulatory--go-straight",
    36: "regulatory--go-straight-or-turn-right",
    37: "regulatory--go-straight-or-turn-left",
    38: "regulatory--keep-right",
    39: "regulatory--keep-left",
    40: "regulatory--roundabout",
}
