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
ANIMATION_SPEED = 30
ROVER_SPEED = 1.5
CAMERA_FOV = 60
CAMERA_RANGE = 200
BLACK_THRESHOLD = 50
SAFE_DISTANCE = 3
WAYPOINT_DISTANCE = 15
GRID_SIZE = 5
MAX_PATH_ATTEMPTS = 5
ROVER_SIZE = 8
TARGET_SIZE = 10

# === SPECIFIC COORDINATES ===
# These are the fixed start and end coordinates
START_COORDINATES = (591, 418)
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

    def create_safe_terrain_mask(self):
        # Create basic safe mask based on terrain brightness
        safe_mask = self.gray_image > BLACK_THRESHOLD
        
        # Apply morphological operations to ensure safe navigation
        kernel = np.ones((SAFE_DISTANCE*2, SAFE_DISTANCE*2), np.uint8)
        safe_mask = cv2.erode(safe_mask.astype(np.uint8), kernel, iterations=1)
        
        # Smooth the mask
        safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_CLOSE, 
                                   np.ones((SAFE_DISTANCE, SAFE_DISTANCE), np.uint8))
        
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
        
        return grid

    def is_position_safe(self, pos):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.safe_mask.shape[1] and 0 <= y < self.safe_mask.shape[0]:
            return self.safe_mask[y, x]
        return False

    def find_nearest_safe_position(self, pos, search_radius=30):
        x, y = int(pos[0]), int(pos[1])
        if self.is_position_safe(pos):
            return pos
        
        # Use spiral search pattern
        for radius in range(1, search_radius, 1):
            for angle in range(0, 360, 15):
                test_x = x + radius * math.cos(math.radians(angle))
                test_y = y + radius * math.sin(math.radians(angle))
                test_x = max(0, min(DISPLAY_WIDTH-1, test_x))
                test_y = max(0, min(DISPLAY_HEIGHT-1, test_y))
                
                if self.is_position_safe((test_x, test_y)):
                    return (int(test_x), int(test_y))
        
        return pos

    def pos_to_grid(self, pos):
        """Convert image position to grid coordinates"""
        x, y = pos
        grid_x = min(int(x // GRID_SIZE), self.grid_width - 1)
        grid_y = min(int(y // GRID_SIZE), self.grid_height - 1)
        return (grid_x, grid_y)

    def grid_to_pos(self, grid_pos):
        """Convert grid coordinates to image position"""
        gx, gy = grid_pos
        x = gx * GRID_SIZE + GRID_SIZE // 2
        y = gy * GRID_SIZE + GRID_SIZE // 2
        return (x, y)

# === A* Path Planner ===
class AStarPathPlanner:
    def __init__(self, terrain_analyzer):
        self.terrain = terrain_analyzer

    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, pos):
        """Get valid neighboring grid positions"""
        x, y = pos
        neighbors = []
        
        # 8-directional movement (including diagonals)
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
        """A* pathfinding algorithm"""
        if not self.terrain.navigation_grid[start_grid[1], start_grid[0]]:
            return None
        if not self.terrain.navigation_grid[goal_grid[1], goal_grid[0]]:
            return None

        frontier = []
        heapq.heappush(frontier, (0, start_grid))
        came_from = {}
        cost_so_far = {start_grid: 0}

        while frontier:
            current_cost, current = heapq.heappop(frontier)

            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                # Calculate movement cost (higher for diagonal moves)
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = 1.414 if (dx + dy == 2) else 1.0
                
                new_cost = cost_so_far[current] + move_cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        return None

    def plan_robust_path(self, start, end):
        """Plan path with multiple fallback strategies"""
        # Convert to safe positions first
        safe_start = self.terrain.find_nearest_safe_position(start)
        safe_end = self.terrain.find_nearest_safe_position(end)
        
        print(f"Planning path from {safe_start} to {safe_end}")
        
        # Try A* pathfinding first
        start_grid = self.terrain.pos_to_grid(safe_start)
        end_grid = self.terrain.pos_to_grid(safe_end)
        
        grid_path = self.a_star_search(start_grid, end_grid)
        
        if grid_path:
            # Convert grid path to image coordinates
            image_path = [self.terrain.grid_to_pos(grid_pos) for grid_pos in grid_path]
            # Smooth the path
            return self.smooth_path(image_path)
        
        # Fallback: Try with different goal positions around the target
        for attempt in range(MAX_PATH_ATTEMPTS):
            angle = (attempt * 360) // MAX_PATH_ATTEMPTS
            offset_x = 20 * math.cos(math.radians(angle))
            offset_y = 20 * math.sin(math.radians(angle))
            
            alt_end = (safe_end[0] + offset_x, safe_end[1] + offset_y)
            alt_end = self.terrain.find_nearest_safe_position(alt_end)
            alt_end_grid = self.terrain.pos_to_grid(alt_end)
            
            grid_path = self.a_star_search(start_grid, alt_end_grid)
            if grid_path:
                image_path = [self.terrain.grid_to_pos(grid_pos) for grid_pos in grid_path]
                # Add final segment to actual target
                image_path.append(safe_end)
                return self.smooth_path(image_path)
        
        # Fallback: Direct path if possible
        if self.is_direct_path_safe(safe_start, safe_end):
            return self.create_smooth_path(safe_start, safe_end, 20)
        
        # Last resort: Edge following
        return self.edge_following_path(safe_start, safe_end)

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

    def is_direct_path_safe(self, start, end, num_checks=100):
        """Check if direct path between two points is safe"""
        for i in range(num_checks + 1):
            t = i / num_checks
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            if not self.terrain.is_position_safe((x, y)):
                return False
        return True

    def edge_following_path(self, start, end):
        """Edge following method for complex terrain"""
        path, current = [start], start
        max_iterations = 1000
        
        for iteration in range(max_iterations):
            angle = math.atan2(end[1]-current[1], end[0]-current[0])
            best_pos, best_dist = None, float('inf')
            
            # Try multiple angles around the target direction
            for a in range(-90, 91, 5):
                t = angle + math.radians(a)
                tx = current[0] + WAYPOINT_DISTANCE * math.cos(t)
                ty = current[1] + WAYPOINT_DISTANCE * math.sin(t)
                
                # Ensure within bounds
                tx = max(BOUNDARY_MARGIN, min(DISPLAY_WIDTH - BOUNDARY_MARGIN, tx))
                ty = max(BOUNDARY_MARGIN, min(DISPLAY_HEIGHT - BOUNDARY_MARGIN, ty))
                
                if self.terrain.is_position_safe((tx, ty)):
                    dist = math.hypot(end[0]-tx, end[1]-ty)
                    if dist < best_dist:
                        best_pos, best_dist = (int(tx), int(ty)), dist
            
            if best_pos:
                current = best_pos
                path.append(current)
                
                # Check if we're close enough to the target
                if best_dist < WAYPOINT_DISTANCE * 1.5:
                    if self.is_direct_path_safe(current, end):
                        path.append(end)
                    break
            else:
                break
        
        return path

    def create_smooth_path(self, start, end, num_points):
        """Create smooth direct path"""
        return [(int(start[0] + t*(end[0]-start[0])), 
                int(start[1] + t*(end[1]-start[1]))) 
                for t in np.linspace(0, 1, num_points)]

# === Enhanced Navigation System ===
class EnhancedRoverNavigationSystem:
    def __init__(self, start_pos, target_pos, terrain_analyzer, camera_estimator):
        self.start_pos = start_pos
        self.target_pos = target_pos
        self.terrain = terrain_analyzer
        self.camera = camera_estimator
        
        # Use enhanced path planner
        path_planner = AStarPathPlanner(terrain_analyzer)
        self.planned_path = path_planner.plan_robust_path(start_pos, target_pos)
        
        self.current_pos = list(start_pos)
        self.path_history = [tuple(start_pos)]
        self.current_waypoint_index = 0
        self.reached_target = False
        self.total_distance_traveled = 0
        self.stuck_counter = 0
        self.last_positions = []

    def update_position(self, speed):
        if self.reached_target:
            return True
            
        if not self.planned_path or self.current_waypoint_index >= len(self.planned_path):
            self.reached_target = True
            return True
        
        # Get current target waypoint
        target = self.planned_path[self.current_waypoint_index]
        dx, dy = target[0] - self.current_pos[0], target[1] - self.current_pos[1]
        dist = math.hypot(dx, dy)
        
        # Check if we're close enough to current waypoint
        if dist <= speed * 1.2:
            self.current_pos = list(target)
            self.path_history.append(tuple(target))
            self.total_distance_traveled += dist
            self.current_waypoint_index += 1
            self.stuck_counter = 0
        else:
            if dist > 0:
                # Move towards current waypoint
                move_x = (dx / dist) * speed
                move_y = (dy / dist) * speed
                new_pos = [self.current_pos[0] + move_x, self.current_pos[1] + move_y]
                
                # Verify new position is safe
                if self.terrain.is_position_safe(new_pos):
                    self.current_pos = new_pos
                    self.path_history.append(tuple(new_pos))
                    self.total_distance_traveled += speed
                    self.stuck_counter = 0
                else:
                    # If new position is unsafe, try to find safe alternative
                    safe_pos = self.terrain.find_nearest_safe_position(new_pos, 15)
                    self.current_pos = list(safe_pos)
                    self.path_history.append(tuple(safe_pos))
                    self.stuck_counter += 1
        
        # Check for stuck condition and replan if necessary
        self.detect_and_handle_stuck()
        
        # Check if reached final destination
        final_dist = math.hypot(self.target_pos[0] - self.current_pos[0], 
                               self.target_pos[1] - self.current_pos[1])
        if final_dist <= speed * 2:
            self.reached_target = True
            return True
            
        return False

    def detect_and_handle_stuck(self):
        """Detect if rover is stuck and replan path"""
        self.last_positions.append(tuple(self.current_pos))
        if len(self.last_positions) > 15:
            self.last_positions.pop(0)
        
        # Check if rover hasn't moved much in recent history
        if len(self.last_positions) >= 15:
            recent_positions = self.last_positions[-8:]
            max_movement = max(math.hypot(pos[0] - self.current_pos[0], 
                                        pos[1] - self.current_pos[1]) 
                             for pos in recent_positions)
            
            if max_movement < 3 or self.stuck_counter > 3:
                print("Rover appears stuck, replanning path...")
                self.replan_path()

    def replan_path(self):
        """Replan path from current position"""
        path_planner = AStarPathPlanner(self.terrain)
        new_path = path_planner.plan_robust_path(self.current_pos, self.target_pos)
        
        if new_path and len(new_path) > 1:
            self.planned_path = new_path
            self.current_waypoint_index = 1  # Skip current position
            self.stuck_counter = 0
            self.last_positions.clear()
            print(f"Replanned path with {len(new_path)} waypoints")

    def get_camera_readings(self):
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
    if not os.path.exists(path): 
        return None
    try:
        moon_tiff = tifffile.imread(path)
        moon_gray = cv2.normalize(moon_tiff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        moon_bgr = cv2.cvtColor(moon_gray, cv2.COLOR_GRAY2BGR)
        return cv2.resize(moon_bgr, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Error loading TIFF file: {e}")
        return None

def create_fallback_moon_terrain():
    """Create a fallback terrain if TIFF cannot be loaded"""
    terrain = np.ones((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8) * 150
    
    # Add some terrain features around the path area
    for _ in range(30):
        x = np.random.randint(50, DISPLAY_WIDTH-50)
        y = np.random.randint(50, DISPLAY_HEIGHT-50)
        radius = np.random.randint(5, 25)
        cv2.circle(terrain, (x, y), radius, (50, 50, 50), -1)
    
    # Ensure start and end points are in safe areas
    cv2.circle(terrain, START_COORDINATES, 30, (200, 200, 200), -1)
    cv2.circle(terrain, END_COORDINATES, 30, (200, 200, 200), -1)
    
    return terrain

def validate_coordinates(coords, name):
    """Validate that coordinates are within bounds"""
    x, y = coords
    if not (0 <= x < DISPLAY_WIDTH and 0 <= y < DISPLAY_HEIGHT):
        print(f"Warning: {name} coordinates {coords} are out of bounds!")
        x = max(0, min(DISPLAY_WIDTH-1, x))
        y = max(0, min(DISPLAY_HEIGHT-1, y))
        print(f"Adjusted {name} coordinates to ({x}, {y})")
        return (x, y)
    return coords

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
        
        # Display terrain
        display_image = self.create_terrain_overlay()
        self.background = self.ax.imshow(display_image, extent=[0, DISPLAY_WIDTH, DISPLAY_HEIGHT, 0])
        
        # Rover and target markers
        self.rover_dot, = self.ax.plot([], [], 'o', markersize=ROVER_SIZE, markerfacecolor='cyan',
                                       markeredgecolor='blue', markeredgewidth=2, label='Rover')
        self.target_dot, = self.ax.plot([], [], 'o', markersize=TARGET_SIZE, markerfacecolor='red',
                                        markeredgecolor='darkred', markeredgewidth=2, label='Target')
        
        # Path lines
        self.path_line, = self.ax.plot([], [], 'g-', linewidth=3, alpha=0.8, label='Actual Path')
        self.planned_path_line, = self.ax.plot([], [], 'y-', linewidth=2, alpha=0.7, 
                                               label='Planned Path', linestyle='--')
        
        # Initialize planned path
        if rover_system.planned_path:
            path_x = [pos[0] for pos in rover_system.planned_path]
            path_y = [pos[1] for pos in rover_system.planned_path]
            self.planned_path_line.set_data(path_x, path_y)
        
        # Set target position
        self.target_dot.set_data([rover_system.target_pos[0]], [rover_system.target_pos[1]])
        
        # Add coordinate labels
        self.ax.annotate(f'START\n{rover_system.start_pos}', 
                        rover_system.start_pos, 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="cyan", alpha=0.7),
                        fontsize=9, fontweight='bold')
        
        self.ax.annotate(f'TARGET\n{rover_system.target_pos}', 
                        rover_system.target_pos, 
                        xytext=(10, -25), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                        fontsize=9, fontweight='bold', color='white')
        
        # Info text
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      verticalalignment='top', fontsize=10,
                                      bbox=dict(boxstyle="round,pad=0.5", facecolor="black", 
                                               alpha=0.9, edgecolor="cyan"),
                                      color='white', fontfamily='monospace')
        
        # Legend and title
        self.ax.legend(loc='upper right', fontsize=10)
        self.ax.set_title(f'Moon Rover Navigation: ({START_COORDINATES[0]},{START_COORDINATES[1]}) → ({END_COORDINATES[0]},{END_COORDINATES[1]})', 
                         fontsize=14, pad=20, fontweight='bold')

    def create_terrain_overlay(self):
        display_img = self.moon_image.copy()
        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        unsafe_mask = gray <= BLACK_THRESHOLD
        display_img[unsafe_mask] = [80, 0, 0]  # Dark red for obstacles
        return cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    def animate(self, frame):
        reached = self.rover_system.update_position(ROVER_SPEED)
        pos = self.rover_system.current_pos
        
        # Update rover position
        self.rover_dot.set_data([pos[0]], [pos[1]])
        
        # Update actual path
        if len(self.rover_system.path_history) > 1:
            path_x = [p[0] for p in self.rover_system.path_history]
            path_y = [p[1] for p in self.rover_system.path_history]
            self.path_line.set_data(path_x, path_y)
        
        # Update planned path if it changed (due to replanning)
        if self.rover_system.planned_path:
            path_x = [pos[0] for pos in self.rover_system.planned_path]
            path_y = [pos[1] for pos in self.rover_system.planned_path]
            self.planned_path_line.set_data(path_x, path_y)
        
        # Update info text
        readings = self.rover_system.get_camera_readings()
        status = 'TARGET REACHED! 🎯' if reached else 'NAVIGATING... 🚀'
        
        self.info_text.set_text(
            f"🌙 MOON ROVER NAVIGATION 🌙\n"
            f"═══════════════════════════\n"
            f"Start: {self.rover_system.start_pos}\n"
            f"Target: {self.rover_system.target_pos}\n"
            f"Current: ({pos[0]:.1f}, {pos[1]:.1f})\n"
            f"Waypoint: {readings['current_waypoint']}/{readings['total_waypoints']}\n"
            f"Distance: {readings['pixel_distance']:.1f} px\n"
            f"Bearing: {readings['bearing_to_target']:.1f}°\n"
            f"Traveled: {self.rover_system.total_distance_traveled:.1f} px\n"
            f"Safe: {'✅' if self.terrain.is_position_safe(pos) else '❌'}\n"
            f"Status: {status}\n"
            f"Frame: {frame}"
        )
        
        return [self.rover_dot, self.path_line, self.planned_path_line, self.info_text]

# === Main Entry Point ===
def main():
    print("=== MOON ROVER NAVIGATION - SPECIFIC COORDINATES ===")
    print(f"Navigating from {START_COORDINATES} to {END_COORDINATES}")
    
    # Validate coordinates
    start_pos = validate_coordinates(START_COORDINATES, "Start")
    target_pos = validate_coordinates(END_COORDINATES, "Target")
    
    # Load moon image
    moon_image = load_moon_image(TIFF_PATH)
    
    if moon_image is None:
        print("Warning: Could not load TIFF file. Using fallback terrain.")
        moon_image = create_fallback_moon_terrain()
    else:
        print(f"Loaded moon terrain: {moon_image.shape}")
    
    # Initialize terrain analyzer
    print("Analyzing terrain for safe navigation...")
    terrain_analyzer = EnhancedTerrainAnalyzer(moon_image)
    
    # Check if start and end positions are safe
    if not terrain_analyzer.is_position_safe(start_pos):
        print(f"Warning: Start position {start_pos} is not safe, finding nearest safe position...")
        start_pos = terrain_analyzer.find_nearest_safe_position(start_pos)
        print(f"Adjusted start position: {start_pos}")
    
    if not terrain_analyzer.is_position_safe(target_pos):
        print(f"Warning: Target position {target_pos} is not safe, finding nearest safe position...")
        target_pos = terrain_analyzer.find_nearest_safe_position(target_pos)
        print(f"Adjusted target position: {target_pos}")
    
    # Calculate straight-line distance
    straight_distance = math.hypot(target_pos[0] - start_pos[0], 
                                   target_pos[1] - start_pos[1])
    print(f"Straight-line distance: {straight_distance:.1f} pixels")
    
    # Initialize camera system
    camera_estimator = CameraDistanceEstimator(
        fov_degrees=CAMERA_FOV,
        known_object_size=TARGET_SIZE,
        focal_length_px=CAMERA_RANGE
    )
    
    # Initialize enhanced navigation system
    print("Initializing enhanced navigation system...")
    rover_nav = EnhancedRoverNavigationSystem(
        start_pos, 
        target_pos, 
        terrain_analyzer, 
        camera_estimator
    )
    
    # Check if path was found
    if not rover_nav.planned_path:
        print("ERROR: Could not find valid path to target!")
        return
    
    print(f"Path planning complete: {len(rover_nav.planned_path)} waypoints")
    
    # Calculate estimated path distance
    # Calculate estimated path distance
    if len(rover_nav.planned_path) > 1:
        path_distance = sum(
            math.hypot(rover_nav.planned_path[i+1][0] - rover_nav.planned_path[i][0],
                      rover_nav.planned_path[i+1][1] - rover_nav.planned_path[i][1])
            for i in range(len(rover_nav.planned_path) - 1)
        )
        print(f"Planned path distance: {path_distance:.1f} pixels")
        print(f"Path efficiency: {(straight_distance/path_distance)*100:.1f}%")
    
    # Initialize visualizer
    print("Starting animation...")
    visualizer = EnhancedRoverAnimationVisualizer(moon_image, rover_nav, terrain_analyzer)
    
    # Create and run animation
    try:
        anim = animation.FuncAnimation(
            visualizer.fig, 
            visualizer.animate, 
            interval=ANIMATION_SPEED,
            blit=False,
            cache_frame_data=False,
            repeat=False
        )
        
        # Show the animation
        plt.tight_layout()
        plt.show()
        
        # Print final statistics
        print("\n=== NAVIGATION COMPLETE ===")
        print(f"Total distance traveled: {rover_nav.total_distance_traveled:.1f} pixels")
        print(f"Straight-line distance: {straight_distance:.1f} pixels")
        if rover_nav.total_distance_traveled > 0:
            efficiency = (straight_distance / rover_nav.total_distance_traveled) * 100
            print(f"Navigation efficiency: {efficiency:.1f}%")
        print(f"Target reached: {'Yes' if rover_nav.reached_target else 'No'}")
        
    except KeyboardInterrupt:
        print("\nAnimation interrupted by user")
    except Exception as e:
        print(f"Animation error: {e}")
        print("Try running without animation or check dependencies")

# === Alternative Static Visualization ===
def create_static_visualization(moon_image, rover_nav, terrain_analyzer):
    """Create a static visualization if animation fails"""
    plt.figure(figsize=(16, 10))
    
    # Display terrain
    display_img = moon_image.copy()
    gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    unsafe_mask = gray <= BLACK_THRESHOLD
    display_img[unsafe_mask] = [80, 0, 0]  # Dark red for obstacles
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(display_img, extent=[0, DISPLAY_WIDTH, DISPLAY_HEIGHT, 0])
    
    # Plot planned path
    if rover_nav.planned_path:
        path_x = [pos[0] for pos in rover_nav.planned_path]
        path_y = [pos[1] for pos in rover_nav.planned_path]
        plt.plot(path_x, path_y, 'y--', linewidth=2, alpha=0.7, label='Planned Path')
    
    # Plot start and end points
    plt.plot(rover_nav.start_pos[0], rover_nav.start_pos[1], 'o', 
             markersize=ROVER_SIZE, markerfacecolor='cyan', 
             markeredgecolor='blue', markeredgewidth=2, label='Start')
    plt.plot(rover_nav.target_pos[0], rover_nav.target_pos[1], 'o', 
             markersize=TARGET_SIZE, markerfacecolor='red', 
             markeredgecolor='darkred', markeredgewidth=2, label='Target')
    
    # Add annotations
    plt.annotate(f'START\n{rover_nav.start_pos}', 
                rover_nav.start_pos, 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="cyan", alpha=0.7),
                fontsize=9, fontweight='bold')
    
    plt.annotate(f'TARGET\n{rover_nav.target_pos}', 
                rover_nav.target_pos, 
                xytext=(10, -25), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                fontsize=9, fontweight='bold', color='white')
    
    plt.xlim(0, DISPLAY_WIDTH)
    plt.ylim(DISPLAY_HEIGHT, 0)
    plt.legend()
    plt.title(f'Moon Rover Navigation Plan: ({START_COORDINATES[0]},{START_COORDINATES[1]}) → ({END_COORDINATES[0]},{END_COORDINATES[1]})', 
              fontsize=14, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.show()

# === Debug Functions ===
def debug_terrain_analysis(terrain_analyzer):
    """Debug function to visualize terrain analysis"""
    print("=== TERRAIN ANALYSIS DEBUG ===")
    
    # Show safe terrain mask
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(terrain_analyzer.gray_image, cmap='gray')
    plt.title('Original Terrain (Grayscale)')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(terrain_analyzer.safe_mask, cmap='RdYlGn')
    plt.title('Safe Terrain Mask')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(terrain_analyzer.navigation_grid, cmap='RdYlGn')
    plt.title('Navigation Grid')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    # Overlay safe areas on original image
    overlay = terrain_analyzer.original_image.copy()
    overlay[~terrain_analyzer.safe_mask] = [255, 0, 0]  # Red for unsafe areas
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Safety Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    total_pixels = terrain_analyzer.safe_mask.size
    safe_pixels = np.sum(terrain_analyzer.safe_mask)
    print(f"Total terrain pixels: {total_pixels}")
    print(f"Safe terrain pixels: {safe_pixels}")
    print(f"Safe terrain percentage: {(safe_pixels/total_pixels)*100:.1f}%")

def run_path_analysis(rover_nav):
    """Analyze the planned path"""
    if not rover_nav.planned_path:
        print("No path to analyze")
        return
    
    print("=== PATH ANALYSIS ===")
    print(f"Number of waypoints: {len(rover_nav.planned_path)}")
    
    # Calculate path segments
    segments = []
    for i in range(len(rover_nav.planned_path) - 1):
        p1 = rover_nav.planned_path[i]
        p2 = rover_nav.planned_path[i + 1]
        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        segments.append((distance, angle))
    
    total_distance = sum(seg[0] for seg in segments)
    print(f"Total path distance: {total_distance:.1f} pixels")
    
    # Analyze path complexity
    direction_changes = 0
    for i in range(1, len(segments)):
        angle_diff = abs(segments[i][1] - segments[i-1][1])
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > 15:  # Significant direction change
            direction_changes += 1
    
    print(f"Direction changes: {direction_changes}")
    print(f"Path complexity: {'High' if direction_changes > len(segments)//3 else 'Medium' if direction_changes > len(segments)//6 else 'Low'}")

# === Enhanced Main Function with Options ===
def main_with_options():
    """Enhanced main function with debugging and visualization options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Moon Rover Navigation System')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualizations')
    parser.add_argument('--static', action='store_true', help='Show static visualization only')
    parser.add_argument('--analyze', action='store_true', help='Run path analysis')
    
    args = parser.parse_args()
    
    # Run main navigation
    main()
    
    if args.debug or args.static or args.analyze:
        # Re-initialize for additional analysis
        moon_image = load_moon_image(TIFF_PATH)
        if moon_image is None:
            moon_image = create_fallback_moon_terrain()
        
        start_pos = validate_coordinates(START_COORDINATES, "Start")
        target_pos = validate_coordinates(END_COORDINATES, "Target")
        
        terrain_analyzer = EnhancedTerrainAnalyzer(moon_image)
        camera_estimator = CameraDistanceEstimator()
        rover_nav = EnhancedRoverNavigationSystem(start_pos, target_pos, terrain_analyzer, camera_estimator)
        
        if args.debug:
            debug_terrain_analysis(terrain_analyzer)
        
        if args.static:
            create_static_visualization(moon_image, rover_nav, terrain_analyzer)
        
        if args.analyze:
            run_path_analysis(rover_nav)

# === Entry Point ===
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("Trying alternative execution...")
        main_with_options()
