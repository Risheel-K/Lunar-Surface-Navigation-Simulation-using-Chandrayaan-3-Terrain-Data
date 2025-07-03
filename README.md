# Lunar-Surface-Navigation-Simulation-using-Chandrayaan-3-Terrain-Data

Overview

This module simulates autonomous rover navigation over lunar terrain using real Chandrayaan-3 elevation data. It demonstrates safe-path planning, crater avoidance, and real-time movement logic purely via software simulation (no hardware used).

Terrain Data Extraction and Preprocessing

    Source: Chandrayaan-3 Digital Terrain Model (DTM) .tif file.

    Original Size: 35,312 × 59,405 pixels (~2.09 billion pixels).

    Resampled Size: 1200 × 800 pixels using bilinear interpolation.

    Processing Steps:

        Intensity thresholding (<50) to identify unsafe zones (craters).

        Morphological closing + erosion to refine boundaries.

        Binary safety mask generated for path planning.

![rawmoon](https://github.com/user-attachments/assets/c72c97c8-599a-4254-a36c-c685725ff8e6)

Processed elevation map with safe vs. unsafe zones
Chandrayaan Terrain Processed Map

Path Planning Strategy

    Algorithm Used: A* (preferred over RRT* for known terrains).

    Justification:

        Deterministic & optimal for static maps.

        Works on 2D navigation grids.

        Avoids overhead from random sampling.


A* vs RRT*

Feature	A*	RRT*
Path Predictability	High (Deterministic)	Variable (Sampling-based)
Performance on Grid Maps	Excellent	Suboptimal
Processing Time	Low	Higher


Rover Simulation Logic

    Start: Random safe point

    Goal: Random reachable safe zone

    Checks:

        Waypoint distance and angle

        Terrain safety at each pixel

    Fallbacks:

        Edge-following

        Goal adjustment if unreachable

![moonstart](https://github.com/user-attachments/assets/987a127d-2eb2-4ea5-842d-6cf638b8f9ad)

Initial stage of lunar navigation simulation. The rover begins at (622, 407)
and navigates toward the target at (765, 525), with planned (dashed yellow) and actual
(green) paths over Chandrayaan-3 terrain data.

![moonhalfpath](https://github.com/user-attachments/assets/6eafa9bd-a8dd-4363-b246-580472fa9a7e)

Rover mid-way through lunar terrain, navigating around crater edges and
obstacles.

![moonreached](https://github.com/user-attachments/assets/8b8bb5a3-9e74-4265-b5f7-b51d845431e2)

Final position of rover after successful path execution
Final Target Reached

Performance Evaluation


Grid Size	5 pixels
Path Planning Time (avg)	< 200 ms
Target Reach Accuracy	> 97% (within 3 pixels)
Path Efficiency	88% – 95%
Obstacle Avoidance Success	100%
Simulation FPS (rendered)	~20 FPS

Conclusion

This module proves the viability of terrain-aware navigation using only elevation data and image processing. It forms a foundation for future embedded lunar rovers by validating:

    A* efficiency in terrain traversal.

    Real-time fallback logic.

    Software-only intelligent path computation over high-risk terrain.
