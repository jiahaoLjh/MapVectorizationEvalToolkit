# eval classes
LINE_CLASSES = {"divider": 0, "boundary": 2}
POLYGON_CLASSES = {"ped_crossing": 1}

# rasterization resolution for computing IoU
WIDTH = 240
HEIGHT = 480

# range of detection in 3D world coordinate
PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

# size of dilation applied before computing IoU
DILATION_LINE = 5
DILATION_POLYGON = 5

# classification threshold
INST_THRESHOLD_LINE = 0.05
INST_THRESHOLD_POLYGON = 0.05

# IoU thresholds
IOUTHRS_LINE = dict(start=0.25, end=0.50, step=0.05)
IOUTHRS_POLYGON = dict(start=0.50, end=0.75, step=0.05)

# number of detections used for evaluation in each frame
MAXDETS_LINE = [1, 10, 100]
MAXDETS_POLYGON = [1, 10, 100]

# evaluation settings
QUERY_LINE = [
    ("AP", "all", "all", 100),
    ("AP", "all", "divider", 100),
    ("AP", "all", "boundary", 100),
    ("AP", 0.25, "all", 100),
    ("AP", 0.25, "divider", 100),
    ("AP", 0.25, "boundary", 100),
    ("AP", 0.50, "all", 100),
    ("AP", 0.50, "divider", 100),
    ("AP", 0.50, "boundary", 100),
    ("AR", "all", "all", 1),
    ("AR", "all", "all", 10),
    ("AR", "all", "all", 100),
    ("AR", "all", "divider", 100),
    ("AR", "all", "boundary", 100),
]
QUERY_POLYGON = [
    ("AP", "all", "ped_crossing", 100),
    ("AP", 0.50, "ped_crossing", 100),
    ("AP", 0.75, "ped_crossing", 100),
    ("AR", "all", "ped_crossing", 1),
    ("AR", "all", "ped_crossing", 10),
    ("AR", "all", "ped_crossing", 100),
]
