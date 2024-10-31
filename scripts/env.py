import math
from typing import List

# Class to represent a 3D point
class Point3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 +
                         (self.y - other.y) ** 2 +
                         (self.z - other.z) ** 2)

    def to_2d(self, plane: str = 'xy'):
        """Converts the 3D point to a 2D point by projecting onto a plane.
        Available planes: 'xy', 'xz', 'yz'
        """
        if plane == 'xy':
            return Point2D(self.x, self.y)
        elif plane == 'xz':
            return Point2D(self.x, self.z)
        elif plane == 'yz':
            return Point2D(self.y, self.z)
        else:
            raise ValueError("Invalid plane specified. Choose 'xy', 'xz', or 'yz'.")

    def __repr__(self):
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"
    
PointCloud3D = List[Point3D]

def create_point_cloud(arrays) -> PointCloud3D:
    """
    Converts a list of objects with size 3 into a PointCloud3D.

    Parameters:
        arrays : List of 3-element objects representing (x, y, z) coordinates.

    Returns:
        PointCloud3D: List of Point3D objects.
    """
    point_cloud = [Point3D(arr[0], arr[1], arr[2]) for arr in arrays if len(arr) == 3]
    return point_cloud


# Class to represent a 2D point
class Point2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 +
                         (self.y - other.y) ** 2)

    def __repr__(self):
        return f"Point2D(x={self.x}, y={self.y})"


# Class to represent a 2D circular obstacle
class CircularObstacle:
    def __init__(self, center: Point2D, radius: float):
        self.center = center
        self.radius = radius

    def is_point_inside(self, point: Point2D) -> bool:
        return self.center.distance_to(point) <= self.radius

    def __repr__(self):
        return f"CircularObstacle(center={self.center}, radius={self.radius})"


# Class to represent an environment that includes obstacles
class Environment2D:
    def __init__(self):
        self.obstacles = []

    def add_obstacle(self, obstacle: CircularObstacle):
        self.obstacles.append(obstacle)

    def is_point_in_obstacle(self, point: Point2D) -> bool:
        for obstacle in self.obstacles:
            if obstacle.is_point_inside(point):
                return True
        return False

    def __repr__(self):
        return f"Environment2D(obstacles={self.obstacles})"