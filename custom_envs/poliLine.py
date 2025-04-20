import numpy as np

from shapely.geometry import LineString
from shapely.geometry import Polygon, Point

def calculate_angle(ray_start, ray_end, seg_start, seg_end):
    ray_vec = np.array([ray_end[0] - ray_start[0], ray_end[1] - ray_start[1]])
    seg_vec = np.array([seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]])
    dot_product = np.dot(ray_vec, seg_vec)
    norm_ray = np.linalg.norm(ray_vec)
    norm_seg = np.linalg.norm(seg_vec)
    return np.arccos(dot_product / (norm_ray * norm_seg))





# Polyline Class
class Polyline:
    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("X and Y vectors must have the same length.")
        self.x = np.array(x)
        self.y = np.array(y)
        self.segments = [LineString([(self.x[i], self.y[i]), (self.x[i + 1], self.y[i + 1])]) 
                    for i in range(len(self.x) - 1)]

    def plot(self):
        import matplotlib.pyplot as plt

        plt.plot(self.x, self.y, "-o", linewidth=2)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Polyline")
        plt.grid(True)
        plt.show()

    def compute_mirrored_points(self, px, py):
        mirror_x = []
        mirror_y = []
        for i in range(len(self.x) - 1):
            x1, y1 = self.x[i], self.y[i]
            x2, y2 = self.x[i + 1], self.y[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            length_sq = dx**2 + dy**2

            t = ((px - x1) * dx + (py - y1) * dy) / length_sq
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy

            mx = 2 * proj_x - px
            my = 2 * proj_y - py

            mirror_x.append(mx)
            mirror_y.append(my)
        return np.array(mirror_x), np.array(mirror_y)

    def check_ray_existence(self, tx_x, tx_y, mirror_x, mirror_y, rx_x, rx_y):
        ray_existence = np.zeros(len(mirror_x), dtype=bool)
        alpha = np.zeros(len(mirror_x))
        for i in range(len(mirror_x)):
            out_ray_x = [mirror_x[i], rx_x]
            out_ray_y = [mirror_y[i], rx_y]

            intersections, segment_indices = self._check_intersection(out_ray_x, out_ray_y)
            if len(intersections) == 1 and segment_indices[0] == i:
                in_ray_x = [tx_x, intersections[0][0]]
                in_ray_y = [tx_y, intersections[0][1]]
                inner_intersections, segment_indices = self._check_intersection(in_ray_x, in_ray_y)
                if len(inner_intersections) == 1 and segment_indices[0] == i:
                    ray_existence[i] = True
                    ray_start = np.array([rx_x, rx_y])
                    ray_end = np.array([mirror_x[i], mirror_y[i]])
                    segment_start = np.array([self.x[i], self.y[i]])
                    segment_end = np.array([self.x[i + 1], self.y[i + 1]])

                    angle_value = calculate_angle(ray_start, ray_end, segment_start, segment_end)
                    alpha[i] = min(
                        angle_value,
                        np.pi
                        - angle_value
                    )
        return ray_existence, alpha

    def check_los_ray_existence(self, tx_x, tx_y, px, py):
        if isinstance(tx_x, float) or isinstance(tx_x, int):
            tx_x = np.array([tx_x])
            tx_y = np.array([tx_y])
        ray_existence = np.ones(len(tx_x), dtype=bool)
        for i in range(len(tx_x)):
            ray_x = [tx_x[i], px]
            ray_y = [tx_y[i], py]
            intersections, segment_indices = self._check_intersection(ray_x, ray_y)
            if len(intersections) >= 1:
                ray_existence[i] = False
        return ray_existence

    def _check_intersection(self, ray_x, ray_y):
        # Initialize lists to store the intersections and corresponding segment indices
        intersections = []
        segment_indices = []

        ray = LineString(zip(ray_x, ray_y))
        # Iterate over the polyline segments and check for intersections with the ray
        for i in range(len(self.x) - 1):
            # Check for intersection between the ray and the current segment
              # Create a LineString for the ray
            intersection = ray.intersection(self.segments[i])
            
            if intersection.is_empty:
                continue  # No intersection with this segment, skip to the next one

            if isinstance(intersection, Point):
                # Single intersection point
                intersections.append((intersection.x, intersection.y))
                segment_indices.append(i)  # Append the current segment index
            elif isinstance(intersection, LineString):
                # If intersection is a LineString (could happen in rare cases)
                for geom in intersection.geoms:
                    intersections.append((geom.x, geom.y))
                    segment_indices.append(i)  # Append the current segment index
            else:
                # Handle multi-part geometries (e.g., geometry collections)
                for geom in intersection.geoms:
                    intersections.append((geom.x, geom.y))
                    segment_indices.append(i)  # Append the current segment index

        return intersections, segment_indices

    def iterate_points(self, accuracy):
        x_min, x_max = np.min(self.x), np.max(self.x)
        y_min, y_max = np.min(self.y), np.max(self.y)
        x_vals = np.arange(x_min, x_max, accuracy)
        y_vals = np.arange(y_min, y_max, accuracy)

        grid_x, grid_y = np.meshgrid(x_vals, y_vals)
        points = np.column_stack([grid_x.flatten(), grid_y.flatten()])

        poly = Polygon(zip(self.x, self.y))
        inside = [Point(p).within(poly) for p in points]
        return points[inside]

