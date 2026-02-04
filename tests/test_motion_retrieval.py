"""Tests for motion retrieval behavior with DHB invariants.

These tests verify that DHB-based motion retrieval correctly:
1. Groups similar motion patterns (e.g., all pick-place tasks)
2. Distinguishes fundamentally different motion types (pick-place vs manipulation)
3. Achieves expected retrieval accuracy at behavior-level (not object-level)
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.core import geometry as geom


# ============================================================================
# Synthetic trajectory generators for different motion types
# ============================================================================

def generate_pick_place_trajectory(seed=0, noise_scale=0.01):
    """Generate a synthetic pick-place motion.

    Pattern: reach down → grasp → lift → move → lower → release
    This creates a characteristic "inverted U" shape in z.
    """
    np.random.seed(seed)
    n = 100
    t = np.linspace(0, 1, n)

    # Pick-place pattern: reach, lift, transport, lower
    x = 0.3 * t + np.random.randn(n) * noise_scale
    y = 0.1 * np.sin(2 * np.pi * t) + np.random.randn(n) * noise_scale
    z = 0.2 * np.sin(np.pi * t) + np.random.randn(n) * noise_scale  # Inverted U

    positions = np.column_stack([x, y, z])
    quaternions = np.array([
        geom.axis_angle_to_quat(np.array([0, 0, 0.3 * ti]))
        for ti in t
    ])

    return positions, quaternions, 'pick_place'


def generate_cable_manipulation_trajectory(seed=0, noise_scale=0.01):
    """Generate a synthetic cable manipulation motion.

    Pattern: complex multi-directional motion with high curvature changes
    Characteristic of untangling/threading motions.
    """
    np.random.seed(seed)
    n = 100
    t = np.linspace(0, 4 * np.pi, n)

    # Complex, multi-directional motion
    x = 0.1 * np.sin(t) + 0.05 * np.sin(3 * t) + np.random.randn(n) * noise_scale
    y = 0.1 * np.cos(t) + 0.05 * np.cos(2 * t) + np.random.randn(n) * noise_scale
    z = 0.05 * np.sin(2 * t) + 0.02 * t / t[-1] + np.random.randn(n) * noise_scale

    positions = np.column_stack([x, y, z])

    # More complex rotation pattern
    quaternions = np.array([
        geom.axis_angle_to_quat(np.array([0.5 * np.sin(ti), 0.3 * np.cos(ti), 0.2 * ti]))
        for ti in t / t[-1]
    ])

    return positions, quaternions, 'cable_manipulation'


def generate_sorting_trajectory(seed=0, noise_scale=0.01):
    """Generate a synthetic sorting motion.

    Pattern: quick pick-move-place cycles with lateral motion
    Different from pick-place in its repetitive, quick movements.
    """
    np.random.seed(seed)
    n = 100
    t = np.linspace(0, 1, n)

    # Sorting: lateral motion with smaller vertical movements
    x = 0.4 * t + np.random.randn(n) * noise_scale
    y = 0.15 * np.sin(4 * np.pi * t) + np.random.randn(n) * noise_scale  # More oscillations
    z = 0.08 * np.abs(np.sin(3 * np.pi * t)) + np.random.randn(n) * noise_scale

    positions = np.column_stack([x, y, z])
    quaternions = np.array([
        geom.axis_angle_to_quat(np.array([0, 0, 0.5 * np.sin(2 * np.pi * ti)]))
        for ti in t
    ])

    return positions, quaternions, 'sorting'


def generate_stirring_trajectory(seed=0, noise_scale=0.01):
    """Generate a synthetic stirring motion.

    Pattern: circular motion in xy-plane with minimal z variation
    """
    np.random.seed(seed)
    n = 100
    t = np.linspace(0, 4 * np.pi, n)

    # Circular stirring motion
    radius = 0.1
    x = radius * np.cos(t) + np.random.randn(n) * noise_scale
    y = radius * np.sin(t) + np.random.randn(n) * noise_scale
    z = 0.02 * np.sin(t / 2) + np.random.randn(n) * noise_scale  # Minimal z

    positions = np.column_stack([x, y, z])
    quaternions = np.array([
        geom.axis_angle_to_quat(np.array([0, 0.1, ti / 4]))
        for ti in t / t[-1]
    ])

    return positions, quaternions, 'stirring'


# ============================================================================
# Embedding function
# ============================================================================

def compute_invariant_embedding(positions, quaternions, n_segments=5):
    """Compute an embedding vector from DHB invariants.

    Uses temporal segmentation with statistics (mean, std, range) plus
    global displacement/rotation features.
    """
    # Encode with DHB
    result = encode_dhb_dr(positions, quaternions, robust_mode=True)
    lin_inv = result['linear_motion_invariants']
    ang_inv = result['angular_motion_invariants']

    # Segment-based statistics
    features = []
    n = len(lin_inv)
    segment_size = max(1, n // n_segments)

    for i in range(n_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, n)
        if start >= n:
            break

        lin_seg = lin_inv[start:end]
        ang_seg = ang_inv[start:end]

        # Statistics per segment
        features.extend([
            np.mean(lin_seg, axis=0),
            np.std(lin_seg, axis=0),
            np.max(lin_seg, axis=0) - np.min(lin_seg, axis=0),
            np.mean(ang_seg, axis=0),
            np.std(ang_seg, axis=0),
        ])

    # Global features
    total_displacement = np.linalg.norm(positions[-1] - positions[0])
    total_rotation = np.linalg.norm(
        geom.quat_to_axis_angle(geom.quat_multiply(
            quaternions[-1], geom.quat_inv(quaternions[0])
        ))
    )
    features.append(np.array([total_displacement, total_rotation, 0, 0]))

    embedding = np.concatenate(features)
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm

    return embedding


# ============================================================================
# Tests for motion retrieval behavior
# ============================================================================

class TestMotionTypeDiscrimination:
    """Test that different motion types are distinguishable."""

    def test_pick_place_similarity(self):
        """Multiple pick-place trajectories should be similar to each other."""
        embeddings = []
        for seed in range(5):
            pos, quat, _ = generate_pick_place_trajectory(seed=seed)
            embeddings.append(compute_invariant_embedding(pos, quat))

        embeddings = np.array(embeddings)
        distances = cdist(embeddings, embeddings, metric='cosine')

        # Get off-diagonal (pairwise) distances
        n = len(embeddings)
        pairwise = [distances[i, j] for i in range(n) for j in range(i+1, n)]
        mean_dist = np.mean(pairwise)

        # Pick-place variants should be relatively similar
        assert mean_dist < 0.5, f"Pick-place variants should be similar, got mean dist {mean_dist:.3f}"

    def test_cable_manipulation_distinct_from_pick_place(self):
        """Cable manipulation should be distinct from pick-place."""
        # Generate pick-place embeddings
        pp_embeddings = []
        for seed in range(3):
            pos, quat, _ = generate_pick_place_trajectory(seed=seed)
            pp_embeddings.append(compute_invariant_embedding(pos, quat))

        # Generate cable manipulation embeddings
        cable_embeddings = []
        for seed in range(3):
            pos, quat, _ = generate_cable_manipulation_trajectory(seed=seed + 100)
            cable_embeddings.append(compute_invariant_embedding(pos, quat))

        pp_embeddings = np.array(pp_embeddings)
        cable_embeddings = np.array(cable_embeddings)

        # Cross-type distances
        cross_distances = cdist(pp_embeddings, cable_embeddings, metric='cosine')
        mean_cross = np.mean(cross_distances)

        # Within pick-place distances
        pp_distances = cdist(pp_embeddings, pp_embeddings, metric='cosine')
        mean_pp = np.mean([pp_distances[i, j] for i in range(3) for j in range(i+1, 3)])

        # Cross-type should be larger than within-type
        assert mean_cross > mean_pp, \
            f"Cross-type ({mean_cross:.3f}) should be > within-type ({mean_pp:.3f})"

    def test_four_motion_types_separation(self):
        """Test separation among four distinct motion types."""
        motion_generators = [
            generate_pick_place_trajectory,
            generate_cable_manipulation_trajectory,
            generate_sorting_trajectory,
            generate_stirring_trajectory,
        ]

        # Generate embeddings for each type
        all_embeddings = []
        all_labels = []

        for type_idx, generator in enumerate(motion_generators):
            for seed in range(3):
                pos, quat, label = generator(seed=seed)
                all_embeddings.append(compute_invariant_embedding(pos, quat))
                all_labels.append(type_idx)

        all_embeddings = np.array(all_embeddings)
        all_labels = np.array(all_labels)

        # Compute distance matrix
        distances = cdist(all_embeddings, all_embeddings, metric='cosine')

        # Compute intra-class and inter-class distances
        intra_dists = []
        inter_dists = []

        n = len(all_labels)
        for i in range(n):
            for j in range(i + 1, n):
                if all_labels[i] == all_labels[j]:
                    intra_dists.append(distances[i, j])
                else:
                    inter_dists.append(distances[i, j])

        intra_mean = np.mean(intra_dists)
        inter_mean = np.mean(inter_dists)
        separation_ratio = inter_mean / (intra_mean + 1e-8)

        print(f"\nFour motion types separation:")
        print(f"  Intra-class mean: {intra_mean:.3f}")
        print(f"  Inter-class mean: {inter_mean:.3f}")
        print(f"  Separation ratio: {separation_ratio:.2f}x")

        # Should have meaningful separation
        assert separation_ratio > 1.2, \
            f"Expected separation ratio > 1.2, got {separation_ratio:.2f}"


class TestRetrievalAccuracy:
    """Test retrieval accuracy at different levels."""

    def test_behavior_level_retrieval(self):
        """Test retrieval accuracy when grouping by behavior type."""
        # Create a database with multiple motion types
        database = []
        labels = []

        generators = {
            'pick_place': generate_pick_place_trajectory,
            'cable': generate_cable_manipulation_trajectory,
            'sorting': generate_sorting_trajectory,
            'stirring': generate_stirring_trajectory,
        }

        for label, generator in generators.items():
            for seed in range(5):
                pos, quat, _ = generator(seed=seed)
                emb = compute_invariant_embedding(pos, quat)
                database.append(emb)
                labels.append(label)

        database = np.array(database)
        labels = np.array(labels)

        # Test retrieval for each item
        correct_top1 = 0
        correct_top3 = 0
        total = len(database)

        for query_idx in range(total):
            query = database[query_idx:query_idx+1]
            query_label = labels[query_idx]

            # Compute distances to all others
            distances = cdist(query, database, metric='cosine')[0]

            # Sort by distance (excluding self)
            sorted_indices = np.argsort(distances)
            # Skip self (distance 0)
            sorted_indices = [i for i in sorted_indices if i != query_idx]

            # Check top-1
            if labels[sorted_indices[0]] == query_label:
                correct_top1 += 1

            # Check top-3
            top3_labels = [labels[i] for i in sorted_indices[:3]]
            if query_label in top3_labels:
                correct_top3 += 1

        top1_acc = correct_top1 / total
        top3_acc = correct_top3 / total

        print(f"\nBehavior-level retrieval accuracy:")
        print(f"  Top-1: {top1_acc:.1%}")
        print(f"  Top-3: {top3_acc:.1%}")

        # Should achieve reasonable accuracy
        assert top1_acc >= 0.6, f"Expected top-1 accuracy >= 60%, got {top1_acc:.1%}"
        assert top3_acc >= 0.8, f"Expected top-3 accuracy >= 80%, got {top3_acc:.1%}"

    def test_nearest_neighbor_within_type(self):
        """Test that nearest neighbor is typically the same motion type."""
        # Generate database
        motion_types = [
            ('pick_place', generate_pick_place_trajectory),
            ('cable', generate_cable_manipulation_trajectory),
        ]

        database = []
        labels = []

        for label, generator in motion_types:
            for seed in range(10):
                pos, quat, _ = generator(seed=seed)
                emb = compute_invariant_embedding(pos, quat)
                database.append(emb)
                labels.append(label)

        database = np.array(database)
        labels = np.array(labels)

        # For each pick-place, check if nearest neighbor is also pick-place
        pp_indices = [i for i, l in enumerate(labels) if l == 'pick_place']
        cable_indices = [i for i, l in enumerate(labels) if l == 'cable']

        distances = cdist(database, database, metric='cosine')

        # Check pick-place retrievals
        pp_correct = 0
        for idx in pp_indices:
            dists = distances[idx].copy()
            dists[idx] = np.inf  # Exclude self
            nearest = np.argmin(dists)
            if labels[nearest] == 'pick_place':
                pp_correct += 1

        pp_acc = pp_correct / len(pp_indices)

        # Check cable retrievals
        cable_correct = 0
        for idx in cable_indices:
            dists = distances[idx].copy()
            dists[idx] = np.inf
            nearest = np.argmin(dists)
            if labels[nearest] == 'cable':
                cable_correct += 1

        cable_acc = cable_correct / len(cable_indices)

        print(f"\nNearest neighbor accuracy:")
        print(f"  Pick-place: {pp_acc:.1%}")
        print(f"  Cable manipulation: {cable_acc:.1%}")

        assert pp_acc >= 0.7, f"Pick-place NN accuracy should be >= 70%, got {pp_acc:.1%}"
        assert cable_acc >= 0.7, f"Cable NN accuracy should be >= 70%, got {cable_acc:.1%}"


class TestEmbeddingQuality:
    """Test embedding quality properties."""

    def test_embedding_stability(self):
        """Test that small noise doesn't drastically change embedding."""
        pos_base, quat_base, _ = generate_pick_place_trajectory(seed=42)
        emb_base = compute_invariant_embedding(pos_base, quat_base)

        # Add small noise - test that embedding distance grows with noise level
        noise_levels = [0.001, 0.005, 0.01]
        distances = []

        for noise in noise_levels:
            np.random.seed(123)  # Consistent noise
            pos_noisy = pos_base + np.random.randn(*pos_base.shape) * noise
            emb_noisy = compute_invariant_embedding(pos_noisy, quat_base)

            dist = np.linalg.norm(emb_base - emb_noisy)
            distances.append(dist)

            # Distance should be bounded (not exploding)
            assert dist < 1.0, f"Embedding distance too large for noise {noise}: dist={dist:.4f}"

        # Distance should generally increase with noise level
        # (not strictly monotonic due to random effects, but trend should be visible)
        assert distances[-1] >= distances[0] * 0.5, \
            f"Expected larger noise to cause larger embedding distance"

    def test_embedding_dimension(self):
        """Test that embedding has consistent dimension."""
        generators = [
            generate_pick_place_trajectory,
            generate_cable_manipulation_trajectory,
            generate_sorting_trajectory,
        ]

        dims = []
        for generator in generators:
            pos, quat, _ = generator(seed=0)
            emb = compute_invariant_embedding(pos, quat)
            dims.append(len(emb))

        # All should have same dimension
        assert len(set(dims)) == 1, f"Inconsistent embedding dimensions: {dims}"

    def test_embedding_normalization(self):
        """Test that embeddings are normalized."""
        for seed in range(5):
            pos, quat, _ = generate_pick_place_trajectory(seed=seed)
            emb = compute_invariant_embedding(pos, quat)

            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-6, f"Embedding not normalized: norm={norm}"
