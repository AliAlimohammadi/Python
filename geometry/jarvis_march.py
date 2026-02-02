"""
Jarvis March (Gift Wrapping) algorithm for finding the convex hull of a set of points.

The convex hull is the smallest convex polygon that contains all the points.

Time Complexity: O(n*h) where n is the number of points and h is the number of
hull points.
Space Complexity: O(h) where h is the number of hull points.

USAGE:
    -> Import this file into your project.
    -> Use the jarvis_march() function to find the convex hull of a set of points.
    -> Parameters:
        -> points: A list of Point objects representing 2D coordinates

REFERENCES:
    -> Wikipedia reference: https://en.wikipedia.org/wiki/Gift_wrapping_algorithm
    -> GeeksforGeeks: https://www.geeksforgeeks.org/convex-hull-set-1-jarviss-algorithm-or-wrapping/
"""

# pytest: disable=pytest-run-parallel

from __future__ import annotations


class Point:
    """
    Represents a 2D point with x and y coordinates.

    >>> p = Point(1.0, 2.0)
    >>> p.x
    1.0
    >>> p.y
    2.0
    """

    def __init__(self, x_coordinate: float, y_coordinate: float) -> None:
        self.x = x_coordinate
        self.y = y_coordinate

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"

    def __hash__(self) -> int:
        return hash((self.x, self.y))


def _cross_product(origin: Point, point_a: Point, point_b: Point) -> float:
    """
    Calculate the cross product of vectors OA and OB.

    Returns:
        > 0: Counter-clockwise turn (left turn)
        = 0: Collinear
        < 0: Clockwise turn (right turn)

    >>> origin = Point(0, 0)
    >>> point_a = Point(1, 1)
    >>> point_b = Point(2, 0)
    >>> _cross_product(origin, point_a, point_b) < 0
    True
    >>> _cross_product(origin, Point(1, 0), Point(2, 0)) == 0
    True
    >>> _cross_product(origin, Point(1, 0), Point(1, 1)) > 0
    True
    """
    return (point_a.x - origin.x) * (point_b.y - origin.y) - (point_a.y - origin.y) * (
        point_b.x - origin.x
    )


def _is_point_on_segment(p1: Point, p2: Point, point: Point) -> bool:
    """
    Check if a point lies on the line segment between p1 and p2.

    >>> _is_point_on_segment(Point(0, 0), Point(2, 2), Point(1, 1))
    True
    >>> _is_point_on_segment(Point(0, 0), Point(2, 2), Point(3, 3))
    False
    >>> _is_point_on_segment(Point(0, 0), Point(2, 0), Point(1, 0))
    True
    """
    # Check if point is collinear with segment endpoints
    cross = (point.y - p1.y) * (p2.x - p1.x) - (point.x - p1.x) * (p2.y - p1.y)

    if abs(cross) > 1e-9:
        return False

    # Check if point is within the bounding box of the segment
    return min(p1.x, p2.x) <= point.x <= max(p1.x, p2.x) and min(
        p1.y, p2.y
    ) <= point.y <= max(p1.y, p2.y)


def jarvis_march(points: list[Point]) -> list[Point]:
    """
    Find the convex hull of a set of points using the Jarvis March algorithm.

    The algorithm starts with the leftmost point and wraps around the set of points,
    selecting the most counter-clockwise point at each step.

    Args:
        points: List of Point objects representing 2D coordinates

    Returns:
        List of Points that form the convex hull in counter-clockwise order.
        Returns empty list if there are fewer than 3 non-collinear points.

    Examples:
        >>> # Triangle
        >>> p1, p2, p3 = Point(1, 1), Point(2, 1), Point(1.5, 2)
        >>> hull = jarvis_march([p1, p2, p3])
        >>> len(hull)
        3
        >>> all(p in hull for p in [p1, p2, p3])
        True

        >>> # Collinear points return empty hull
        >>> points = [Point(i, 0) for i in range(5)]
        >>> jarvis_march(points)
        []

        >>> # Rectangle with interior point - interior point excluded
        >>> p1, p2 = Point(1, 1), Point(2, 1)
        >>> p3, p4 = Point(2, 2), Point(1, 2)
        >>> p5 = Point(1.5, 1.5)
        >>> hull = jarvis_march([p1, p2, p3, p4, p5])
        >>> len(hull)
        4
        >>> p5 in hull
        False

        >>> # Star shape - only tips are in hull
        >>> tips = [
        ...     Point(-5, 6), Point(-11, 0), Point(-9, -8),
        ...     Point(4, 4), Point(6, -7)
        ... ]
        >>> interior = [Point(-7, -2), Point(-2, -4), Point(0, 1)]
        >>> hull = jarvis_march(tips + interior)
        >>> len(hull)
        5
        >>> all(p in hull for p in tips)
        True
        >>> any(p in hull for p in interior)
        False

        >>> # Too few points
        >>> jarvis_march([])
        []
        >>> jarvis_march([Point(0, 0)])
        []
        >>> jarvis_march([Point(0, 0), Point(1, 1)])
        []
    """
    if len(points) <= 2:
        return []

    convex_hull: list[Point] = []

    # Find the leftmost point (and bottom-most in case of tie)
    left_point_idx = 0
    for i in range(1, len(points)):
        if points[i].x < points[left_point_idx].x or (
            points[i].x == points[left_point_idx].x
            and points[i].y < points[left_point_idx].y
        ):
            left_point_idx = i

    convex_hull.append(Point(points[left_point_idx].x, points[left_point_idx].y))

    current_idx = left_point_idx
    while True:
        # Find the next counter-clockwise point
        next_idx = (current_idx + 1) % len(points)
        for i in range(len(points)):
            if _cross_product(points[current_idx], points[i], points[next_idx]) > 0:
                next_idx = i

        if next_idx == left_point_idx:
            # Completed constructing the hull
            break

        current_idx = next_idx

        # Check if the last point is collinear with new point and second-to-last
        last = len(convex_hull) - 1
        if len(convex_hull) > 1 and _is_point_on_segment(
            convex_hull[last - 1], convex_hull[last], points[current_idx]
        ):
            # Remove the last point from the hull
            convex_hull[last] = Point(points[current_idx].x, points[current_idx].y)
        else:
            convex_hull.append(Point(points[current_idx].x, points[current_idx].y))

    # Check for edge case: last point collinear with first and second-to-last
    if len(convex_hull) <= 2:
        return []

    last = len(convex_hull) - 1
    if _is_point_on_segment(convex_hull[last - 1], convex_hull[last], convex_hull[0]):
        convex_hull.pop()
        if len(convex_hull) == 2:
            return []

    return convex_hull


if __name__ == "__main__":
    import doctest

    doctest.testmod()
