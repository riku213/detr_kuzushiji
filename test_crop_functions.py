#!/usr/bin/env python
"""Test script for crop functionality."""

from datasets.kuzushiji_text import (
    compute_crop_grid,
    bbox_intersects_crop,
    transform_bbox_to_crop_coords,
)

print("Testing crop utility functions...")

# Test compute_crop_grid
crops = compute_crop_grid(1000, 1000, 4)
print(f"\n✓ Grid cells: {len(crops)}")
print(f"  First crop: {crops[0]}")
print(f"  Last crop: {crops[-1]}")
assert len(crops) == 16, "Expected 16 crop cells for 4x4 grid"

# Test bbox_intersects_crop
crop_box = (0, 0, 250, 250)
intersects = bbox_intersects_crop(100, 100, 200, 200, crop_box)
print(f"\n✓ BBox (100,100,200,200) intersects crop (0,0,250,250): {intersects}")
assert intersects, "Expected intersection"

# Test non-intersecting bbox
no_intersect = bbox_intersects_crop(300, 300, 100, 100, crop_box)
print(f"✓ BBox (300,300,100,100) intersects crop (0,0,250,250): {no_intersect}")
assert not no_intersect, "Expected no intersection"

# Test transform_bbox_to_crop_coords
cx, cy, w, h = transform_bbox_to_crop_coords(100, 100, 50, 50, crop_box, 250, 250)
print(f"\n✓ Transformed bbox coords:")
print(f"  Center: ({cx:.4f}, {cy:.4f})")
print(f"  Size: ({w:.4f}, {h:.4f})")
assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0, "Center should be in [0,1]"
assert 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0, "Size should be in [0,1]"

print("\n✅ All utility functions working correctly!")
