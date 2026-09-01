import numpy as np

# ============================================================
# ROOM DEFINITIONS (anticlockwise)
# ============================================================

# V-Trap Room
# (7,10) -> (14,10) -> (14,20) -> (7,20)
V_ROOM = [
    (7.0, 10.0),
    (14.0, 10.0),
    (14.0, 20.0),
    (7.0, 20.0)
]

# Gap-Trap Room
# (7,0) -> (14,0) -> (14,10) -> (7,10)
GAP_ROOM = [
    (7.0, 0.0),
    (14.0, 0.0),
    (14.0, 10.0),
    (7.0, 10.0)
]

# ============================================================
# OBSTACLE PARAMETERS
# ============================================================

V_RADIUS = 0.25      # V-trap obstacle radius
GAP_RADIUS = 0.25    # Gap-trap obstacle radius

# Gap-trap clearance
CLEAR_GAP = 0.27

np.random.seed(42)

# ============================================================
# V-TRAPS (TOP ROOM)
# ============================================================

def generate_v_traps():
    traps = []

    # Room limits
    xmin, xmax = 7.0, 14.0
    ymin, ymax = 10.0, 20.0

    # Same layout as HTML but shifted into room
    x_coords = [1.5, 5.0]
    y_coords = [2.5, 7.0, 11.5]

    angle_rad = np.radians(40)

    # Touching 0.20m radius obstacles
    dist_step = 2 * V_RADIUS  # 0.40m

    for y in y_coords:
        for x in x_coords:

            alpha = np.random.uniform(0, 2 * np.pi)

            p0 = [round(x, 3), round(y, 3)]

            p1 = [
                round(x + dist_step * np.cos(alpha + angle_rad), 3),
                round(y + dist_step * np.sin(alpha + angle_rad), 3)
            ]

            p2 = [
                round(x + 2 * dist_step * np.cos(alpha + angle_rad), 3),
                round(y + 2 * dist_step * np.sin(alpha + angle_rad), 3)
            ]

            p3 = [
                round(x + dist_step * np.cos(alpha - angle_rad), 3),
                round(y + dist_step * np.sin(alpha - angle_rad), 3)
            ]

            p4 = [
                round(x + 2 * dist_step * np.cos(alpha - angle_rad), 3),
                round(y + 2 * dist_step * np.sin(alpha - angle_rad), 3)
            ]
            
            traps.append([p0, p1, p2, p3, p4])

    return traps


# ============================================================
# GAP TRAPS (BOTTOM ROOM)
# ============================================================

def generate_gap_traps():
    traps = []

    # Room limits
    xmin, xmax = 7.0, 14.0
    ymin, ymax = 0.0, 10.0

    # Same layout as HTML but shifted into lower room
    # x_coords = [1.0, 3.5, 6.0]
    # y_coords = [2.5, 5.5, 9.0, 12.0]
    x_coords = [8.0, 10.5, 12.0]
    y_coords = [1.5, 4.5, 7.3, 10.0, 12.5]

    # Center-to-center distance
    # 0.25 + 0.25 + 0.27 = 0.77 m
    dist_cc = 2 * GAP_RADIUS + CLEAR_GAP

    for y in y_coords:
        for x in x_coords:

            beta = np.random.uniform(0, 2 * np.pi)

            p1 = [
                round(x - (dist_cc / 2) * np.cos(beta), 3),
                round(y - (dist_cc / 2) * np.sin(beta), 3)
            ]

            p2 = [
                round(x + (dist_cc / 2) * np.cos(beta), 3),
                round(y + (dist_cc / 2) * np.sin(beta), 3)
            ]

            traps.append([p1, p2])

    return traps


# ============================================================
# GENERATE
# ============================================================

v_traps = generate_v_traps()
gap_traps = generate_gap_traps()

# ============================================================
# PRINT
# ============================================================

print("V_TRAPS = [")
for i, trap in enumerate(v_traps):
    print(f"    {trap},  # Trap {i+1}")
print("]")

print("\nGAP_TRAPS = [")
for i, trap in enumerate(gap_traps):
    print(f"    {trap},  # Trap {i+1}")
print("]")