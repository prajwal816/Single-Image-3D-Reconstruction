"""
Simulated ShapeNet Dataset.

Generates synthetic RGB images (rendered-style views of 3D primitives)
and corresponding 3D point clouds for training single-image 3D reconstruction.

Supports categories: cube, sphere, cylinder, cone, torus.
Procedural generation with deterministic seeding per sample for reproducibility.
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from typing import Dict, List, Optional, Tuple


# ---- 3D primitive point cloud generators ---------------------------

def _sample_cube(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample points uniformly on the surface of a unit cube centred at origin."""
    points = []
    pts_per_face = max(n // 6, 1)
    for axis in range(3):
        for sign in [-0.5, 0.5]:
            uv = rng.uniform(-0.5, 0.5, size=(pts_per_face, 2))
            face = np.zeros((pts_per_face, 3))
            dims = [d for d in range(3) if d != axis]
            face[:, dims[0]] = uv[:, 0]
            face[:, dims[1]] = uv[:, 1]
            face[:, axis] = sign
            points.append(face)
    points = np.concatenate(points, axis=0)
    idx = rng.choice(len(points), size=n, replace=len(points) < n)
    return points[idx].astype(np.float32)


def _sample_sphere(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample points uniformly on a unit sphere."""
    phi = rng.uniform(0, 2 * math.pi, n)
    cos_theta = rng.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)
    x = 0.5 * np.sin(theta) * np.cos(phi)
    y = 0.5 * np.sin(theta) * np.sin(phi)
    z = 0.5 * np.cos(theta)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _sample_cylinder(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample points on a unit cylinder (radius=0.5, height=1)."""
    # 70% on curved surface, 15% each cap
    n_side = int(0.7 * n)
    n_cap = (n - n_side) // 2
    n_cap2 = n - n_side - n_cap

    # Side
    theta_side = rng.uniform(0, 2 * math.pi, n_side)
    h_side = rng.uniform(-0.5, 0.5, n_side)
    x_side = 0.5 * np.cos(theta_side)
    y_side = 0.5 * np.sin(theta_side)
    side = np.stack([x_side, y_side, h_side], axis=-1)

    caps = []
    for z_val, nc in [(-0.5, n_cap), (0.5, n_cap2)]:
        r = 0.5 * np.sqrt(rng.uniform(0, 1, nc))
        theta_c = rng.uniform(0, 2 * math.pi, nc)
        cap = np.stack([r * np.cos(theta_c), r * np.sin(theta_c),
                        np.full(nc, z_val)], axis=-1)
        caps.append(cap)

    return np.concatenate([side] + caps, axis=0).astype(np.float32)


def _sample_cone(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample points on a cone (apex at top, base radius=0.5, height=1)."""
    n_side = int(0.7 * n)
    n_base = n - n_side

    # Side surface
    t = rng.uniform(0, 1, n_side)  # height parameter
    theta = rng.uniform(0, 2 * math.pi, n_side)
    r = 0.5 * (1.0 - t)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = t - 0.5  # range: -0.5 to 0.5
    side = np.stack([x, y, z], axis=-1)

    # Base
    r_base = 0.5 * np.sqrt(rng.uniform(0, 1, n_base))
    theta_b = rng.uniform(0, 2 * math.pi, n_base)
    base = np.stack([r_base * np.cos(theta_b), r_base * np.sin(theta_b),
                     np.full(n_base, -0.5)], axis=-1)

    return np.concatenate([side, base], axis=0).astype(np.float32)


def _sample_torus(n: int, rng: np.random.RandomState, R: float = 0.35,
                  r: float = 0.15) -> np.ndarray:
    """Sample points on a torus (major R, minor r)."""
    theta = rng.uniform(0, 2 * math.pi, n)
    phi = rng.uniform(0, 2 * math.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


PRIMITIVE_SAMPLERS = {
    "cube": _sample_cube,
    "sphere": _sample_sphere,
    "cylinder": _sample_cylinder,
    "cone": _sample_cone,
    "torus": _sample_torus,
}


# ---- Synthetic image rendering ------------------------------------

_CATEGORY_COLORS = {
    "cube":     ((40, 120, 210), (20, 60, 130)),
    "sphere":   ((220, 80, 60),  (140, 40, 30)),
    "cylinder": ((60, 190, 90),  (30, 120, 50)),
    "cone":     ((240, 180, 40), (160, 110, 20)),
    "torus":    ((180, 60, 200), (100, 30, 120)),
}


def _render_cube(draw: ImageDraw.Draw, cx: float, cy: float,
                 size: float, colors: tuple, rotation: float):
    """Render an isometric-style cube."""
    s = size * 0.45
    # Front face
    front = [
        (cx - s, cy), (cx, cy + s * 0.6),
        (cx + s, cy), (cx, cy - s * 0.6)
    ]
    draw.polygon(front, fill=colors[0], outline=(255, 255, 255))
    # Top face
    top = [
        (cx - s, cy), (cx, cy - s * 0.6),
        (cx + s * 0.3, cy - s * 0.9), (cx - s * 0.7, cy - s * 0.3)
    ]
    r, g, b = colors[0]
    draw.polygon(top, fill=(min(r + 40, 255), min(g + 40, 255), min(b + 40, 255)))
    # Right face
    right = [
        (cx, cy + s * 0.6), (cx + s, cy),
        (cx + s * 0.3, cy - s * 0.9), (cx + s * 0.3, cy - s * 0.3)
    ]
    draw.polygon(right, fill=colors[1])


def _render_sphere(draw: ImageDraw.Draw, cx: float, cy: float,
                   size: float, colors: tuple, rotation: float):
    """Render a shaded sphere."""
    r = size * 0.4
    bbox = [cx - r, cy - r, cx + r, cy + r]
    draw.ellipse(bbox, fill=colors[0], outline=colors[1])
    # Highlight
    hr = r * 0.25
    draw.ellipse([cx - r * 0.3 - hr, cy - r * 0.3 - hr,
                  cx - r * 0.3 + hr, cy - r * 0.3 + hr],
                 fill=(255, 255, 255, 180))


def _render_cylinder(draw: ImageDraw.Draw, cx: float, cy: float,
                     size: float, colors: tuple, rotation: float):
    """Render a cylinder with top ellipse."""
    w = size * 0.35
    h = size * 0.55
    ew = w  # ellipse width
    eh = size * 0.12  # ellipse height
    # Body
    draw.rectangle([cx - w, cy - h * 0.5, cx + w, cy + h * 0.5], fill=colors[0])
    # Bottom ellipse
    draw.ellipse([cx - ew, cy + h * 0.5 - eh, cx + ew, cy + h * 0.5 + eh],
                 fill=colors[1])
    # Top ellipse
    r, g, b = colors[0]
    draw.ellipse([cx - ew, cy - h * 0.5 - eh, cx + ew, cy - h * 0.5 + eh],
                 fill=(min(r + 30, 255), min(g + 30, 255), min(b + 30, 255)),
                 outline=(255, 255, 255))


def _render_cone(draw: ImageDraw.Draw, cx: float, cy: float,
                 size: float, colors: tuple, rotation: float):
    """Render a cone with base ellipse."""
    w = size * 0.4
    h = size * 0.55
    eh = size * 0.1
    # Triangle body
    draw.polygon([(cx, cy - h * 0.5), (cx - w, cy + h * 0.4),
                  (cx + w, cy + h * 0.4)], fill=colors[0], outline=colors[1])
    # Base ellipse
    draw.ellipse([cx - w, cy + h * 0.4 - eh, cx + w, cy + h * 0.4 + eh],
                 fill=colors[1])


def _render_torus(draw: ImageDraw.Draw, cx: float, cy: float,
                  size: float, colors: tuple, rotation: float):
    """Render a torus (two overlapping ellipses)."""
    rx = size * 0.4
    ry = size * 0.2
    t = size * 0.1
    # Outer ellipse
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=colors[0],
                 outline=colors[1], width=2)
    # Inner hole
    draw.ellipse([cx - rx * 0.45, cy - ry * 0.45,
                  cx + rx * 0.45, cy + ry * 0.45],
                 fill=(30, 30, 35))


PRIMITIVE_RENDERERS = {
    "cube": _render_cube,
    "sphere": _render_sphere,
    "cylinder": _render_cylinder,
    "cone": _render_cone,
    "torus": _render_torus,
}


def _generate_synthetic_image(category: str, rng: np.random.RandomState,
                              image_size: int = 224) -> np.ndarray:
    """Generate a synthetic rendered-style image of a 3D primitive."""
    # Background with subtle gradient
    bg_base = rng.randint(20, 45)
    img = Image.new("RGB", (image_size, image_size), (bg_base, bg_base, bg_base + 5))
    draw = ImageDraw.Draw(img)

    cx = image_size / 2 + rng.uniform(-image_size * 0.05, image_size * 0.05)
    cy = image_size / 2 + rng.uniform(-image_size * 0.05, image_size * 0.05)
    obj_size = image_size * rng.uniform(0.55, 0.75)
    rotation = rng.uniform(0, 2 * math.pi)

    colors = _CATEGORY_COLORS.get(category, ((128, 128, 128), (80, 80, 80)))
    renderer = PRIMITIVE_RENDERERS[category]
    renderer(draw, cx, cy, obj_size, colors, rotation)

    return np.array(img, dtype=np.uint8)


# ---- Dataset class ------------------------------------------------

class SimulatedShapeNetDataset(Dataset):
    """
    Procedurally generates paired (RGB image, 3D point cloud) samples
    for training single-image 3D reconstruction.

    Each sample is deterministically generated from the sample index,
    enabling reproducible training across runs.

    Args:
        num_samples: Total number of samples.
        num_points:  Number of points per point cloud.
        image_size:  Side length of generated square images.
        categories:  List of shape categories to include.
        transform:   Optional torchvision transform for images.
        point_transform: Optional transform for point clouds.
    """

    def __init__(
        self,
        num_samples: int = 50000,
        num_points: int = 1024,
        image_size: int = 224,
        categories: Optional[List[str]] = None,
        transform=None,
        point_transform=None,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_points = num_points
        self.image_size = image_size
        self.categories = categories or list(PRIMITIVE_SAMPLERS.keys())
        self.transform = transform
        self.point_transform = point_transform
        self.seed = seed

        # Pre-assign category per sample for balanced distribution
        rng = np.random.RandomState(seed)
        self.sample_categories = [
            self.categories[i % len(self.categories)]
            for i in range(num_samples)
        ]
        rng.shuffle(self.sample_categories)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.RandomState(self.seed + idx)
        category = self.sample_categories[idx]
        category_id = self.categories.index(category)

        # Generate point cloud
        sampler = PRIMITIVE_SAMPLERS[category]
        point_cloud = sampler(self.num_points, rng)

        # Apply random rotation to point cloud
        angle = rng.uniform(0, 2 * math.pi)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot = np.array([[cos_a, -sin_a, 0],
                        [sin_a,  cos_a, 0],
                        [0,      0,     1]], dtype=np.float32)
        point_cloud = point_cloud @ rot.T

        # Generate corresponding image
        image = _generate_synthetic_image(category, rng, self.image_size)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        point_cloud_tensor = torch.from_numpy(point_cloud)

        # Apply transforms
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if self.point_transform is not None:
            point_cloud_tensor = self.point_transform(point_cloud_tensor)

        return {
            "image": image_tensor,
            "point_cloud": point_cloud_tensor,
            "category_id": torch.tensor(category_id, dtype=torch.long),
            "category_name": category,
            "sample_idx": idx,
        }

    def get_categories(self) -> List[str]:
        """Return list of available categories."""
        return list(self.categories)

    def get_category_counts(self) -> Dict[str, int]:
        """Return number of samples per category."""
        from collections import Counter
        return dict(Counter(self.sample_categories))
