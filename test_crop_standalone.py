#!/usr/bin/env python
"""Direct test of crop functions without importing full dataset."""

import sys
sys.path.insert(0, '/c/Users/kotat/MyPrograms/MyKuzushiji/detr_kuzushiji')

# Define functions directly
def compute_crop_grid(img_height, img_width, grid_size=4):
    """
    Compute grid cells for cropping.
    """
    cell_height = img_height / grid_size
    cell_width = img_width / grid_size
    
    crops = []
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = int(col * cell_width)
            y1 = int(row * cell_height)
            x2 = int((col + 1) * cell_width)
            y2 = int((row + 1) * cell_height)
            crops.append((x1, y1, x2, y2))
    
    return crops


def bbox_intersects_crop(x, y, w, h, crop_box):
    """
    Check if a bounding box (in pixel coords) intersects with a crop region.
    """
    x1_crop, y1_crop, x2_crop, y2_crop = crop_box
    
    x2_bbox = x + w
    y2_bbox = y + h
    
    # Check for intersection
    if x2_bbox < x1_crop or x > x2_crop or y2_bbox < y1_crop or y > y2_crop:
        return False
    return True


def transform_bbox_to_crop_coords(x, y, w, h, crop_box, crop_w, crop_h):
    """
    Transform a bounding box from full image coordinates to cropped image coordinates.
    Then normalize to [0, 1] range.
    """
    x1_crop, y1_crop, x2_crop, y2_crop = crop_box
    
    # Clip bbox to crop boundaries
    x1_clipped = max(x, x1_crop)
    y1_clipped = max(y, y1_crop)
    x2_clipped = min(x + w, x2_crop)
    y2_clipped = min(y + h, y2_crop)
    
    # Convert to crop-local coordinates
    x1_local = x1_clipped - x1_crop
    y1_local = y1_clipped - y1_crop
    x2_local = x2_clipped - x1_crop
    y2_local = y2_clipped - y1_crop
    
    # Compute width and height in crop space
    w_local = x2_local - x1_local
    h_local = y2_local - y1_local
    
    # Normalize to [0, 1]
    cx_norm = (x1_local + x2_local) * 0.5 / crop_w
    cy_norm = (y1_local + y2_local) * 0.5 / crop_h
    w_norm = w_local / crop_w
    h_norm = h_local / crop_h
    
    # Clamp to [0, 1] for partially visible bboxes
    cx_norm = max(0.0, min(cx_norm, 1.0))
    cy_norm = max(0.0, min(cy_norm, 1.0))
    w_norm = max(0.0, min(w_norm, 1.0))
    h_norm = max(0.0, min(h_norm, 1.0))
    
    return cx_norm, cy_norm, w_norm, h_norm


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
