import cv2
import numpy as np
import math
import tifffile
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
from collections import defaultdict

# === SETTINGS ===
TIFF_PATH = r"C:\Users\Risheel K\Desktop\Desktop\projects\isro\data\derived\20090610\ch1_tmc_ndn_20090610T1952322479_d_dtm_d18.tif"
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 800
BOUNDARY_MARGIN = 30
ANIMATION_SPEED = 50  # Reduced for better performance
ROVER_SPEED = 2.0  # Increased speed
CAMERA_FOV = 60
CAMERA_RANGE = 200
BLACK_THRESHOLD = 50
SAFE_DISTANCE = 2  # Reduced for better pathfinding
WAYPOINT_DISTANCE = 10  # Reduced for more precise navigation
GRID_SIZE = 3  # Smaller grid for better precision
MAX_PATH_ATTEMPTS = 8
ROVER_SIZE = 10
TARGET_SIZE = 12
DESTINATION_THRESHOLD = 8.0  # Distance threshold to consider target reached

# === SPECIFIC COORDINATES ===
START_COORDINATES = (622, 407)
END_COORDINATES = (765, 525)

# === Enhanced Terrain Analyzer ===
class EnhancedTerrainAnalyzer:
    def __init__(self, moon_image):
        self.original_image = moon_image.copy()
        self.gray_image = cv2.cvtColor(moon_image, cv2.COLOR_BGR2GRAY)
        self.safe_mask = self.create_safe_terrain_mask()
        self.grid_width = DISPLAY_WIDTH // GRID_SIZE
        self.grid_height = DISPLAY_HEIGHT // GRID_SIZE
        self.navigation_grid = self.create_navigation_grid()
        print(f"Navigation grid created: {self.grid_width}x{self.grid_height}")

    def create_safe_terrain_mask(self):
        # Create basic safe mask based on terrain brightness
        safe_mask = self.gray_image > BLACK_THRESHOLD
        
        # Apply lighter morphological operations to preserve more safe areas
        kernel = np.ones((SAFE_DISTANCE, SAFE_DISTANCE), np.uint8)
        safe_mask = cv2.erode(safe_mask.astype(np.uint8), kernel, iterations=1)
        
        # Fill small holes
        safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_CLOSE, 
                                   np.ones((3, 3), np.uint8))
        
        return safe_mask.astype(bool)

    def create_navigation_grid(self):
        """Create a high-resolution grid for A* pathfinding"""
        grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Convert grid coordinates to image coordinates
                img_x = min(x * GRID_SIZE + GRID_SIZE//2, DISPLAY_WIDTH-1)
                img_y = min(y * GRID_SIZE + GRID_SIZE//2, DISPLAY_HEIGHT-1)
                
                # Check if this grid cell is safe
                grid[y, x] = self.is_position_safe((img_x, img_y))
        
        safe_cells = np.sum(grid)
        total_cells = grid.size
        print(f"Navigation grid: {safe_cells}/{total_cells} safe cells ({safe_cells/total_cells*100:.1f}%)")
        
        return grid

    def is_position_safe(self, pos):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.safe_mask.shape[1] and 0 <= y < self.safe_mask.shape[0]:
            return self.safe_mask[y, x]
        return False

    def find_nearest_safe_position(self, pos, search_radius=50):
        x, y = int(pos[0]), int(pos[1])
        if self.is_position_safe(pos):
            return pos
        
        print(f"Finding safe position near ({x}, {y})")
        
        # Use spiral search pattern with more points
        for radius in range(1, search_radius, 2):
            for angle in range(0, 360, 10):
                test_x = x + radius * math.cos(math.radians(angle))
                test_y = y + radius * math.sin(math.radians(angle))
                test_x = max(BOUNDARY_MARGIN, min(DISPLAY_WIDTH-BOUNDARY_MARGIN, test_x))
                test_y = max(BOUNDARY_MARGIN, min(DISPLAY_HEIGHT-BOUNDARY_MARGIN, test_y))
                
                if self.is_position_safe((test_x, test_y)):
                    print(f"Found safe position at ({int(test_x)}, {int(test_y)})")
                    return (int(test_x), int(test_y))
        
        print(f"Warning: Could not find safe position near ({x}, {y})")
        return pos

    def pos_to_grid(self, pos):
        """Convert image position to grid coordinates"""
        x, y = pos
        grid_x = min(max(0, int(x // GRID_SIZE)), self.grid_width - 1)
        grid_y = min(max(0, int(y // GRID_SIZE)), self.grid_height - 1)
        return (grid_x, grid_y)

    def grid_to_pos(self, grid_pos):
        """Convert grid coordinates to image position"""
        gx, gy = grid_pos
        x = gx * GRID_SIZE + GRID_SIZE // 2
        y = gy * GRID_SIZE + GRID_SIZE // 2
        return (x, y)

# === Improved A* Path Planner ===
class AStarPathPlanner:
    def __init__(self, terrain_analyzer):
        self.terrain = terrain_analyzer

    def heuristic(self, a, b):
        """Manhattan distance heuristic for better performance"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        """Get valid neighboring grid positions"""
        x, y = pos
        neighbors = []
        
        # 8-directional movement
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.terrain.grid_width and 
                0 <= ny < self.terrain.grid_height and
                self.terrain.navigation_grid[ny, nx]):
                neighbors.append((nx, ny))
        
        return neighbors

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from A* search"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def a_star_search(self, start_grid, goal_grid):
        """Enhanced A* pathfinding algorithm"""
        print(f"A* search from {start_grid} to {goal_grid}")
        
        if not (0 <= start_grid[0] < self.terrain.grid_width and 
                0 <= start_grid[1] < self.terrain.grid_height):
            print(f"Start position {start_grid} out of bounds")
            return None
            
        if not (0 <= goal_grid[0] < self.terrain.grid_width and 
                0 <= goal_grid[1] < self.terrain.grid_height):
            print(f"Goal position {goal_grid} out of bounds")
            return None
            
        if not self.terrain.navigation_grid[start_grid[1], start_grid[0]]:
            print(f"Start position {start_grid} is not safe")
            return None
            
        if not self.terrain.navigation_grid[goal_grid[1], goal_grid[0]]:
            print(f"Goal position {goal_grid} is not safe")
            return None

        frontier = []
        heapq.heappush(frontier, (0, start_grid))
        came_from = {}
        cost_so_far = {start_grid: 0}
        nodes_explored = 0

        while frontier and nodes_explored < 10000:  # Limit search to prevent infinite loops
            current_cost, current = heapq.heappop(frontier)
            nodes_explored += 1

            if current == goal_grid:
                path = self.reconstruct_path(came_from, current)
                print(f"A* found path with {len(path)} waypoints after exploring {nodes_explored} nodes")
                return path

            for neighbor in self.get_neighbors(current):
                # Calculate movement cost
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = 1.414 if (dx + dy == 2) else 1.0
                
                new_cost = cost_so_far[current] + move_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        print(f"A* search failed after exploring {nodes_explored} nodes")
        return None

    def plan_robust_path(self, start, end):
        """Plan path with multiple fallback strategies"""
        print(f"Planning robust path from {start} to {end}")
        
        # Convert to safe positions first
        safe_start = self.terrain.find_nearest_safe_position(start)
        safe_end = self.terrain.find_nearest_safe_position(end)
        
        print(f"Safe positions: {safe_start} -> {safe_end}")
        
        # Convert to grid coordinates
        start_grid = self.terrain.pos_to_grid(safe_start)
        end_grid = self.terrain.pos_to_grid(safe_end)
        
        print(f"Grid coordinates: {start_grid} -> {end_grid}")
        
        # Try A* pathfinding first
        grid_path = self.a_star_search(start_grid, end_grid)
        
        if grid_path:
            # Convert grid path to image coordinates
            image_path = [self.terrain.grid_to_pos(grid_pos) for grid_pos in grid_path]
            smoothed_path = self.smooth_path(image_path)
            print(f"Path planned successfully with {len(smoothed_path)} waypoints")
            return smoothed_path
        
        # Fallback strategies
        print("A* failed, trying fallback strategies...")
        
        # Try intermediate waypoints
        for attempt in range(MAX_PATH_ATTEMPTS):
            # Try different intermediate points
            mid_x = safe_start[0] + (safe_end[0] - safe_start[0]) * (attempt + 1) / (MAX_PATH_ATTEMPTS + 1)
            mid_y = safe_start[1] + (safe_end[1] - safe_start[1]) * (attempt + 1) / (MAX_PATH_ATTEMPTS + 1)
            mid_point = self.terrain.find_nearest_safe_position((mid_x, mid_y))
            
            mid_grid = self.terrain.pos_to_grid(mid_point)
            
            # Try path to intermediate point
            path1 = self.a_star_search(start_grid, mid_grid)
            if path1:
                path2 = self.a_star_search(mid_grid, end_grid)
                if path2:
                    # Combine paths
                    combined_grid_path = path1 + path2[1:]  # Remove duplicate middle point
                    image_path = [self.terrain.grid_to_pos(grid_pos) for grid_pos in combined_grid_path]
                    smoothed_path = self.smooth_path(image_path)
                    print(f"Fallback path found via intermediate point with {len(smoothed_path)} waypoints")
                    return smoothed_path
        
        # Last resort: straight line if possible
        if self.is_direct_path_safe(safe_start, safe_end):
            print("Using direct path as last resort")
            return [safe_start, safe_end]
        
        print("All pathfinding strategies failed!")
        return None

    def smooth_path(self, path):
        """Smooth the path by removing unnecessary waypoints"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Look ahead to find the farthest point we can reach directly
            farthest = i + 1
            for j in range(i + 2, len(path)):
                if self.is_direct_path_safe(path[i], path[j]):
                    farthest = j
                else:
                    break
            
            smoothed.append(path[farthest])
            i = farthest
        
        return smoothed

    def is_direct_path_safe(self, start, end, num_checks=50):
        """Check if direct path between two points is safe"""
        for i in range(num_checks + 1):
            t = i / num_checks
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            if not self.terrain.is_position_safe((x, y)):
                return False
        return True

# === Fixed Navigation System ===
class EnhancedRoverNavigationSystem:
    def __init__(self, start_pos, target_pos, terrain_analyzer, camera_estimator):
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.terrain = terrain_analyzer
        self.camera = camera_estimator
        
        # Use enhanced path planner
        path_planner = AStarPathPlanner(terrain_analyzer)
        self.planned_path = path_planner.plan_robust_path(start_pos, target_pos)
        
        if not self.planned_path:
            print("ERROR: No path could be planned!")
            self.planned_path = [start_pos, target_pos]  # Fallback direct path
        
        self.current_pos = list(start_pos)
        self.path_history = [tuple(start_pos)]
        self.current_waypoint_index = 1  # Start from second waypoint (first is start position)
        self.reached_target = False
        self.total_distance_traveled = 0
        self.stuck_counter = 0
        self.last_positions = []
        
        print(f"Navigation system initialized with {len(self.planned_path)} waypoints")

    def update_position(self, speed):
        """Updated position update logic with better target detection"""
        if self.reached_target:
            return True
        
        # Check if we're close enough to the final target
        final_distance = math.hypot(self.target_pos[0] - self.current_pos[0], 
                                   self.target_pos[1] - self.current_pos[1])
        
        if final_distance <= DESTINATION_THRESHOLD:
            print(f"TARGET REACHED! Final distance: {final_distance:.1f}")
            self.reached_target = True
            self.current_pos = list(self.target_pos)
            self.path_history.append(tuple(self.target_pos))
            return True
        
        # Check if we have waypoints to follow
        if self.current_waypoint_index >= len(self.planned_path):
            # No more waypoints, try to go directly to target
            if final_distance <= speed * 2:
                self.current_pos = list(self.target_pos)
                self.path_history.append(tuple(self.target_pos))
                self.reached_target = True
                return True
            else:
                # Move towards target directly
                dx = self.target_pos[0] - self.current_pos[0]
                dy = self.target_pos[1] - self.current_pos[1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    move_x = (dx / dist) * speed
                    move_y = (dy / dist) * speed
                    new_pos = [self.current_pos[0] + move_x, self.current_pos[1] + move_y]
                    
                    if self.terrain.is_position_safe(new_pos):
                        self.current_pos = new_pos
                        self.path_history.append(tuple(new_pos))
                        self.total_distance_traveled += speed
                    else:
                        self.stuck_counter += 1
                        if self.stuck_counter > 5:
                            self.replan_path()
                return False
        
        # Get current target waypoint
        target_waypoint = self.planned_path[self.current_waypoint_index]
        dx = target_waypoint[0] - self.current_pos[0]
        dy = target_waypoint[1] - self.current_pos[1]
        waypoint_dist = math.hypot(dx, dy)
        
        # Check if we've reached the current waypoint
        if waypoint_dist <= speed * 1.5:
            self.current_pos = list(target_waypoint)
            self.path_history.append(tuple(target_waypoint))
            self.total_distance_traveled += waypoint_dist
            self.current_waypoint_index += 1
            self.stuck_counter = 0
            print(f"Reached waypoint {self.current_waypoint_index}/{len(self.planned_path)}")
        else:
            # Move towards current waypoint
            if waypoint_dist > 0:
                move_x = (dx / waypoint_dist) * speed
                move_y = (dy / waypoint_dist) * speed
                new_pos = [self.current_pos[0] + move_x, self.current_pos[1] + move_y]
                
                # Verify new position is safe
                if self.terrain.is_position_safe(new_pos):
                    self.current_pos = new_pos
                    self.path_history.append(tuple(new_pos))
                    self.total_distance_traveled += speed
                    self.stuck_counter = 0
                else:
                    # Try to find a safe alternative nearby
                    safe_pos = self.terrain.find_nearest_safe_position(new_pos, 10)
                    if safe_pos != new_pos:  # Found a different safe position
                        self.current_pos = list(safe_pos)
                        self.path_history.append(tuple(safe_pos))
                        self.total_distance_traveled += math.hypot(safe_pos[0] - self.current_pos[0], 
                                                                 safe_pos[1] - self.current_pos[1])
                    else:
                        self.stuck_counter += 1
        
        # Handle stuck situations
        if self.stuck_counter > 3:
            print("Rover appears stuck, attempting to replan...")
            self.replan_path()
        
        return False

    def replan_path(self):
        """Replan path from current position"""
        print(f"Replanning path from {self.current_pos} to {self.target_pos}")
        
        path_planner = AStarPathPlanner(self.terrain)
        new_path = path_planner.plan_robust_path(self.current_pos, self.target_pos)
        
        if new_path and len(new_path) > 1:
            self.planned_path = new_path
            self.current_waypoint_index = 1  # Skip current position
            self.stuck_counter = 0
            self.last_positions.clear()
            print(f"Replanned path with {len(new_path)} waypoints")
        else:
            print("Replan failed, trying direct movement")
            self.stuck_counter = 0  # Reset to prevent infinite replanning

    def get_camera_readings(self):
        """Get camera readings for display"""
        pdist = self.camera.estimate_distance_by_position(self.current_pos, self.target_pos)
        return {
            'pixel_distance': pdist,
            'apparent_target_size': self.camera.get_apparent_target_size(pdist),
            'bearing_to_target': math.degrees(math.atan2(self.target_pos[1]-self.current_pos[1], 
                                                        self.target_pos[0]-self.current_pos[0])),
            'current_waypoint': self.current_waypoint_index,
            'total_waypoints': len(self.planned_path) if self.planned_path else 0
        }

# === Distance Estimator ===
class CameraDistanceEstimator:
    def __init__(self, fov_degrees=60, known_object_size=20, focal_length_px=500):
        self.fov = math.radians(fov_degrees)
        self.known_object_size = known_object_size
        self.focal_length = focal_length_px

    def estimate_distance_by_position(self, rover_pos, target_pos):
        return math.hypot(target_pos[0] - rover_pos[0], target_pos[1] - rover_pos[1])

    def get_apparent_target_size(self, actual_distance):
        return max((self.known_object_size * self.focal_length) / actual_distance if actual_distance else 1, 1)

# === Utility Functions ===
def load_moon_image(path):
    """Load moon image with better error handling"""
    if not os.path.exists(path): 
        print(f"TIFF file not found at: {path}")
        return None
    try:
        print(f"Loading TIFF file: {path}")
        moon_tiff = tifffile.imread(path)
        print(f"Original TIFF shape: {moon_tiff.shape}, dtype: {moon_tiff.dtype}")
        
        # Handle different bit depths
        if moon_tiff.dtype == np.uint16:
            moon_normalized = cv2.normalize(moon_tiff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            moon_normalized = moon_tiff.astype(np.uint8)
        
        moon_bgr = cv2.cvtColor(moon_normalized, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(moon_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
        print(f"Processed image shape: {resized.shape}")
        return resized
    except Exception as e:
        print(f"Error loading TIFF file: {e}")
        return None

def create_fallback_moon_terrain():
    """Create a more realistic fallback terrain"""
    print("Creating fallback moon terrain...")
    terrain = np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8) * 120
    
    # Add craters and obstacles
    np.random.seed(42)  # For reproducible terrain
    for _ in range(40):
        x = np.random.randint(50, DISPLAY_WIDTH-50)
        y = np.random.randint(50, DISPLAY_HEIGHT-50)
        radius = np.random.randint(8, 35)
        cv2.circle(terrain, (x, y), radius, (30, 30, 30), -1)
    
    # Ensure start and end areas are safe
    cv2.circle(terrain, START_COORDINATES, 40, (180, 180, 180), -1)
    cv2.circle(terrain, END_COORDINATES, 40, (180, 180, 180), -1)
    
    # Add some safe corridors
    cv2.line(terrain, START_COORDINATES, END_COORDINATES, (150, 150, 150), 20)
    
    return terrain

def validate_coordinates(coords, name):
    """Validate and adjust coordinates"""
    x, y = coords
    original_coords = coords
    
    x = max(BOUNDARY_MARGIN, min(DISPLAY_WIDTH - BOUNDARY_MARGIN, x))
    y = max(BOUNDARY_MARGIN, min(DISPLAY_HEIGHT - BOUNDARY_MARGIN, y))
    
    if (x, y) != original_coords:
        print(f"Adjusted {name} coordinates from {original_coords} to ({x}, {y})")
    
    return (x, y)

# === Enhanced Visualizer ===
class EnhancedRoverAnimationVisualizer:
    def __init__(self, moon_image, rover_system, terrain_analyzer):
        self.moon_image = moon_image.copy()
        self.rover_system = rover_system
        self.terrain = terrain_analyzer
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.ax.set_xlim(0, DISPLAY_WIDTH)
        self.ax.set_ylim(DISPLAY_HEIGHT, 0)
        self.ax.set_aspect('equal')
        
        # Display terrain with safety overlay
        display_image = self.create_terrain_overlay()
        self.background = self.ax.imshow(display_image, extent=[0, DISPLAY_WIDTH, DISPLAY_HEIGHT, 0])
        
        # Initialize plot elements
        self.rover_dot, = self.ax.plot([], [], 'o', markersize=ROVER_SIZE, markerfacecolor='cyan',
                                       markeredgecolor='blue', markeredgewidth=2, label='Rover')
        self.target_dot, = self.ax.plot([], [], 'o', markersize=TARGET_SIZE, markerfacecolor='red',
                                        markeredgecolor='darkred', markeredgewidth=2, label='Target')
        
        # Path lines
        self.path_line, = self.ax.plot([], [], 'g-', linewidth=3, alpha=0.9, label='Actual Path')
        self.planned_path_line, = self.ax.plot([], [], 'y-', linewidth=2, alpha=0.7, 
                                               label='Planned Path', linestyle='--')
        
        # Set initial positions
        self.target_dot.set_data([rover_system.target_pos[0]], [rover_system.target_pos[1]])
        
        # Initialize planned path
        if rover_system.planned_path:
            path_x = [pos[0] for pos in rover_system.planned_path]
            path_y = [pos[1] for pos in rover_system.planned_path]
            self.planned_path_line.set_data(path_x, path_y)
        
        # Add annotations
        self.ax.annotate(f'START\n{rover_system.start_pos}', 
                        rover_system.start_pos, 
                        xytext=(15, 15), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="cyan", alpha=0.8),
                        fontsize=10, fontweight='bold')
        
        self.ax.annotate(f'TARGET\n{rover_system.target_pos}', 
                        rover_system.target_pos, 
                        xytext=(15, -30), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
                        fontsize=10, fontweight='bold', color='white')
        
        # Info text
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      verticalalignment='top', fontsize=10,
                                      bbox=dict(boxstyle="round,pad=0.5", facecolor="black", 
                                               alpha=0.9, edgecolor="cyan"),
                                      color='white', fontfamily='monospace')
        
        # Legend and title
        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.set_title(f'Moon Rover Navigation: {START_COORDINATES} → {END_COORDINATES}', 
                         fontsize=14, pad=20, fontweight='bold')
        
        # Add grid for better visualization
        self.ax.grid(True, alpha=0.3)

    def create_terrain_overlay(self):
        """Create terrain overlay with safety visualization"""
        display_img = self.moon_image.copy()
        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        
        # Highlight unsafe areas in red
        unsafe_mask = gray <= BLACK_THRESHOLD
        display_img[unsafe_mask] = [100, 0, 0]  # Dark red for obstacles
        
        # Highlight safe areas slightly
        safe_mask = gray > BLACK_THRESHOLD + 20
        display_img[safe_mask] = np.minimum(display_img[safe_mask] + [0, 20, 0], 255)
        
        return cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    def animate(self, frame):
        """Animation function with improved updates"""
        # Update rover position
        reached = self.rover_system.update_position(ROVER_SPEED)
        pos = self.rover_system.current_pos
        
        # Update rover marker
        self.rover_dot.set_data([pos[0]], [pos[1]])
        
        # Update actual path
        if len(self.rover_system.path_history) > 1:
            path_x = [p[0] for p in self.rover_system.path_history]
            path_y = [p[1] for p in self.rover_system.path_history]
            self.path_line.set_data(path_x, path_y)
        
        # Update planned path if it changed
        if hasattr(self.rover_system, 'planned_path') and self.rover_system.planned_path:
            path_x = [pos[0] for pos in self.rover_system.planned_path]
            path_y = [pos[1] for pos in self.rover_system.planned_path]
            self.planned_path_line.set_data(path_x, path_y)
        
        # Update info display
        readings = self.rover_system.get_camera_readings()
        final_distance = math.hypot(self.rover_system.target_pos[0] - pos[0], 
                                   self.rover_system.target_pos[1] - pos[1])
        
        status = '🎯 TARGET REACHED!' if reached else '🚀 NAVIGATING...'
        
        info_text = f"""MOON ROVER NAVIGATION STATUS
{status}

Position: ({pos[0]:.1f}, {pos[1]:.1f})
Target: {self.rover_system.target_pos}
Distance to Target: {final_distance:.1f} pixels

Waypoint: {readings['current_waypoint']}/{readings['total_waypoints']}
Bearing: {readings['bearing_to_target']:.1f}°
Distance Traveled: {self.rover_system.total_distance_traveled:.1f}

Path Points: {len(self.rover_system.path_history)}
Stuck Counter: {self.rover_system.stuck_counter}
"""
        
        self.info_text.set_text(info_text)
        
        # Stop animation if target reached
        if reached:
            print(f"\n{'='*50}")
            print("🎯 MISSION ACCOMPLISHED! 🎯")
            print(f"Total distance traveled: {self.rover_system.total_distance_traveled:.1f} pixels")
            print(f"Path points recorded: {len(self.rover_system.path_history)}")
            print(f"Final position: ({pos[0]:.1f}, {pos[1]:.1f})")
            print(f"Target position: {self.rover_system.target_pos}")
            print(f"Final distance to target: {final_distance:.1f} pixels")
            print(f"{'='*50}")
        
        return [self.rover_dot, self.target_dot, self.path_line, self.planned_path_line, self.info_text]

    def start_animation(self):
        """Start the animation"""
        print("Starting rover navigation animation...")
        self.animation = animation.FuncAnimation(
            self.fig, self.animate, frames=10000, interval=ANIMATION_SPEED,
            blit=True, repeat=False, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()

# === Main Execution ===
def main():
    """Enhanced main function with comprehensive error handling"""
    print("="*60)
    print("🌙 ENHANCED MOON ROVER NAVIGATION SYSTEM 🚀")
    print("="*60)
    
    # Load moon terrain
    moon_image = load_moon_image(TIFF_PATH)
    if moon_image is None:
        print("Using fallback terrain generation...")
        moon_image = create_fallback_moon_terrain()
    
    # Validate coordinates
    start_pos = validate_coordinates(START_COORDINATES, "START")
    end_pos = validate_coordinates(END_COORDINATES, "END")
    
    print(f"Start position: {start_pos}")
    print(f"End position: {end_pos}")
    print(f"Direct distance: {math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]):.1f} pixels")
    
    # Initialize systems
    print("\nInitializing navigation systems...")
    terrain_analyzer = EnhancedTerrainAnalyzer(moon_image)
    camera_estimator = CameraDistanceEstimator()
    
    # Verify positions are safe
    if not terrain_analyzer.is_position_safe(start_pos):
        print("Warning: Start position is not safe, finding alternative...")
        start_pos = terrain_analyzer.find_nearest_safe_position(start_pos)
        print(f"Adjusted start position: {start_pos}")
    
    if not terrain_analyzer.is_position_safe(end_pos):
        print("Warning: End position is not safe, finding alternative...")
        end_pos = terrain_analyzer.find_nearest_safe_position(end_pos)
        print(f"Adjusted end position: {end_pos}")
    
    # Create navigation system
    rover_system = EnhancedRoverNavigationSystem(
        start_pos, end_pos, terrain_analyzer, camera_estimator
    )
    
    if not rover_system.planned_path:
        print("ERROR: Could not plan any path! Check terrain and coordinates.")
        return
    
    print(f"Path planning completed: {len(rover_system.planned_path)} waypoints")
    for i, waypoint in enumerate(rover_system.planned_path):
        print(f"  Waypoint {i}: {waypoint}")
    
    # Create and start visualization
    print("\nLaunching visualization...")
    visualizer = EnhancedRoverAnimationVisualizer(moon_image, rover_system, terrain_analyzer)
    
    # Print navigation settings
    print(f"\nNavigation Settings:")
    print(f"  Rover Speed: {ROVER_SPEED} pixels/frame")
    print(f"  Grid Size: {GRID_SIZE}x{GRID_SIZE} pixels")
    print(f"  Safe Distance: {SAFE_DISTANCE} pixels") 
    print(f"  Destination Threshold: {DESTINATION_THRESHOLD} pixels")
    print(f"  Animation Speed: {ANIMATION_SPEED}ms per frame")
    
    try:
        visualizer.start_animation()
    except KeyboardInterrupt:
        print("\nAnimation interrupted by user.")
    except Exception as e:
        print(f"Animation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
