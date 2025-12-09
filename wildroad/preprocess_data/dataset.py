import os
import time
import pickle
import argparse
from typing import Any, Dict

import numpy as np
import addict
# import rtree  # replaced by NumpySortedBoxIndex
import scipy
import igraph as ig

import graph_utils


# NumPy-based spatial index (faster than rtree with better multi-threading)
class NumpySortedBoxIndex:
    """Axis-aligned bounding-box index using NumPy searchsorted over x.

    - Build: O(N log N). Stores stable sort order by x and the corresponding sorted x values.
    - Query: O(log N + k). Binary-search on x range, then filter y in bbox.

    This class provides an `intersection((xmin, ymin, xmax, ymax)) -> List[int]` API
    compatible with the subset we used from rtree.
    """

    def __init__(self, points: np.ndarray, order: np.ndarray = None, sorted_x: np.ndarray = None):
        points = np.asarray(points)
        assert points.ndim == 2 and points.shape[1] == 2, "points must be [N,2]"
        self.points = points.astype(np.float32, copy=False)
        if order is None:
            order = np.argsort(self.points[:, 0], kind='mergesort')
            sorted_x = self.points[order, 0]
        # Keep compact dtypes
        self.order = np.asarray(order, dtype=np.int32, order='C')
        self.sorted_x = np.asarray(sorted_x, dtype=np.float32, order='C')

    def intersection(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        # Binary search on x
        left = np.searchsorted(self.sorted_x, xmin, side='left')
        right = np.searchsorted(self.sorted_x, xmax, side='right')
        if right <= left:
            return []
        candidate_idx = self.order[left:right]
        pts = self.points[candidate_idx]
        # Filter on y (and recheck x to be extra safe for float boundaries)
        mask = (
            (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
        )
        return candidate_idx[mask].tolist()

    def to_state(self):
        return {
            'order': self.order.astype(np.int32, copy=False),
            'sorted_x': self.sorted_x.astype(np.float32, copy=False),
        }

class GraphLabelGenerator:
    def __init__(self, config, full_graph, coord_transform):
        self.config = config
        # full_graph: sat2graph format
        # coord_transform: lambda, [N, 2] array -> [N, 2] array
        # convert to igraph for high performance
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(full_graph, coord_transform)
        # find crossover points, we'll avoid predicting these as keypoints
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)  # [(x, y), (),]
        # subdivide version
        self.subdivide_resolution = config.SUBDIVIDE_RESOLUTION
        self.full_graph_subdivide = graph_utils.subdivide_graph(
            self.full_graph_origin, self.subdivide_resolution
        )
        # np array, maybe faster
        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])
        # pre-build spatial index
        # rtree for box queries
        # self.graph_rtee = rtree.index.Index()
        # for i, v in enumerate(self.subdivide_points):
        #     x, y = v
        #     # hack to insert single points
        #     self.graph_rtee.insert(i, (x, y, x, y))
        # numpy-based AABB index (drop-in replacement)
        self.graph_rtee = NumpySortedBoxIndex(self.subdivide_points)
        # kdtree for spherical query
        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

        # pre-exclude points near crossover points
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        # Find intersection points, these will always be kept in nms
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num,), dtype=np.float32)
        self.nms_score_override[np.array(list(itsc_indices))] = 2.0  # itsc points will always be kept

        # Points near crossover and intersections are interesting.
        # they will be more frequently sampled
        interesting_indices = set()
        interesting_radius = config.INTERESTING_RADIUS
        # near itsc
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(np.array(p), interesting_radius)
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num,), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = config.INTR_SAMPLE_WEIGHT

    def sample_patch(self, patch, rot_index=0):
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices  # the point indices in the patch

        # Use NMS to downsample, params shall resemble inference time
        patch_indices = np.array(list(patch_indices))
        if len(patch_indices) == 0:
            sample_num = self.config.TOPO_SAMPLE_NUM
            max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = ([[0, 0]] * max_nbr_queries, [False] * max_nbr_queries, [False] * max_nbr_queries)
            return fake_points, [fake_sample] * sample_num

        patch_points = self.subdivide_points[patch_indices, :]

        # random scores to emulate different random configurations that all share a similar spacing
        # raise scores for intersction points so they are always kept
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)
        nms_radius = self.config.ROAD_NMS_RADIUS

        # kept_indces are into the patch_points array
        nmsed_points, kept_indices = graph_utils.nms_points(
            patch_points, nms_scores, radius=nms_radius, return_indices=True
        )
        # now this is into the subdivide graph
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]

        sample_num = self.config.TOPO_SAMPLE_NUM  # has to be greater than 1
        sample_weights = self.sample_weights[nmsed_indices]
        # indices into the nmsed points in the patch
        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num,
            replace=True,
            p=sample_weights / np.sum(sample_weights),
        )
        # indices into the subdivided graph
        sample_indices = nmsed_indices[sample_indices_in_nmsed]

        radius = self.config.NEIGHBOR_RADIUS
        max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES  # has to be greater than 1
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        # [n_sample, n_nbr]
        # k+1 because the nearest one is always self
        knn_d, knn_idx = nmsed_kdtree.query(
            sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius
        )
        # knn_d inf 填充, knn_idx num_points + 1 填充
        samples = []

        for i in range(sample_num):  # 512
            source_node = sample_indices[i]
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num]
            valid_nbr_indices = valid_nbr_indices[1:]  # the nearest one is self so remove
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]

            # BFS to find immediate neighbors on graph
            reached_nodes = graph_utils.bfs_with_conditions(
                self.full_graph_subdivide, source_node, set(target_nodes), radius // self.subdivide_resolution
            )
            shall_connect = [t in reached_nodes for t in target_nodes]

            pairs = []
            valid = []
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx in valid_nbr_indices:
                pairs.append((source_nmsed_idx, target_nmsed_idx))
                valid.append(True)

            # zero-pad
            for i in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid.append(False)

            samples.append((pairs, shall_connect, valid))

        # Transform points
        # [N, 2]
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        # homo for rot
        # [N, 3]
        nmsed_points = np.concatenate(
            [nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1
        )
        trans = np.array(
            [
                [1, 0, -0.5 * self.config.PATCH_SIZE],
                [0, 1, -0.5 * self.config.PATCH_SIZE],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        # ccw 90 deg in img (x, y)
        rot = np.array(
            [
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )  # 先平移原点到中心，再绕中心旋转
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]

        # Add noise
        noise_scale = self.config.NOISE_SCALE  # pixels
        nmsed_points += np.random.normal(0.0, noise_scale, size=nmsed_points.shape)

        return nmsed_points, samples

    def save_precomputed(self, tile_dir: str) -> None:
        """Persist complex state for ultra-fast reload.

        - Saves igraph.Graph directly via pickle
        - Saves NumPy AABB index arrays (order, sorted_x) into glg_state.pkl
        - Saves numpy arrays and sets without conversions
        """
        os.makedirs(tile_dir, exist_ok=True)

        # Build index state for persistence (NumpySortedBoxIndex)
        index_state = self.graph_rtee.to_state()

        # Save full state including igraph
        state: Dict[str, Any] = {}
        state['version'] = 3
        state['subdivide_points'] = self.subdivide_points.astype(np.float32, copy=False)
        state['exclude_indices'] = self.exclude_indices  # store as set
        state['nms_score_override'] = self.nms_score_override.astype(np.float32, copy=False)
        state['sample_weights'] = self.sample_weights.astype(np.float32, copy=False)
        state['subdivide_resolution'] = int(self.subdivide_resolution)
        state['full_graph_subdivide'] = self.full_graph_subdivide  # igraph.Graph
        # state['rtree_path'] = rtree_base  # deprecated
        # persist numpy AABB index
        state['numpy_index_order'] = index_state['order']
        state['numpy_index_sorted_x'] = index_state['sorted_x']
        state['precompute_config'] = dict(
            SUBDIVIDE_RESOLUTION=int(self.subdivide_resolution),
            INTERESTING_RADIUS=float(self.config.INTERESTING_RADIUS),
            INTR_SAMPLE_WEIGHT=float(self.config.INTR_SAMPLE_WEIGHT),
        )
        with open(os.path.join(tile_dir, 'glg_state.pkl'), 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_precomputed_dir(cls, tile_dir: str, config):
        """Load precomputed GLG without reconstruction (fast path)."""
        state_path = os.path.join(tile_dir, 'glg_state.pkl')
        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        obj = cls.__new__(cls)
        obj.config = config

        obj.subdivide_points = state['subdivide_points']
        obj.exclude_indices = state['exclude_indices']  # already a set
        obj.nms_score_override = state['nms_score_override']
        obj.sample_weights = state['sample_weights']
        obj.subdivide_resolution = int(state['subdivide_resolution'])
        obj.full_graph_subdivide = state['full_graph_subdivide']  # igraph.Graph
        obj.full_graph_origin = None

        # Build numpy AABB index directly
        order = state.get('numpy_index_order')
        sorted_x = state.get('numpy_index_sorted_x')
        if order is not None and sorted_x is not None:
            obj.graph_rtee = NumpySortedBoxIndex(obj.subdivide_points, order=order, sorted_x=sorted_x)
        else:
            # fallback: build on the fly if precompute was done with older version
            obj.graph_rtee = NumpySortedBoxIndex(obj.subdivide_points)
        # No need for global KDTree at runtime here

        return obj


def build_default_config_globalscale() -> addict.Dict:
    """Default config for precompute. Training can override at load time.

    Only SUBDIVIDE_RESOLUTION / INTERESTING_RADIUS / INTR_SAMPLE_WEIGHT
    affect the precomputation stored in files.
    """

    # globalscale config

    cfg = addict.Dict()
    # patch-related params (not used for precompute, kept for completeness)
    cfg.PATCH_SIZE = 512
    cfg.ROAD_NMS_RADIUS = 16
    cfg.TOPO_SAMPLE_NUM = 512
    cfg.NEIGHBOR_RADIUS = 64
    cfg.MAX_NEIGHBOR_QUERIES = 16
    cfg.NOISE_SCALE = 1.0

    # critical for precompute
    cfg.SUBDIVIDE_RESOLUTION = 4
    cfg.INTERESTING_RADIUS = 32
    cfg.INTR_SAMPLE_WEIGHT = 0.9
    return cfg


def build_default_config_wild_road() -> addict.Dict:
    """Default config for precompute. Training can override at load time.

    Only SUBDIVIDE_RESOLUTION / INTERESTING_RADIUS / INTR_SAMPLE_WEIGHT
    affect the precomputation stored in files.
    """

    # wildroad config

    cfg = addict.Dict()
    # patch-related params (not used for precompute, kept for completeness)
    cfg.PATCH_SIZE = 1024
    cfg.ROAD_NMS_RADIUS = 50
    cfg.TOPO_SAMPLE_NUM = 128
    cfg.NEIGHBOR_RADIUS = 200
    cfg.MAX_NEIGHBOR_QUERIES = 8
    cfg.NOISE_SCALE = 3.0

    # critical for precompute
    cfg.SUBDIVIDE_RESOLUTION = 20
    cfg.INTERESTING_RADIUS = 100
    cfg.INTR_SAMPLE_WEIGHT = 1.0
    return cfg


def precompute_and_save_range(start_id: int, end_id: int, output_dir: str, input_dir: str, overwrite: bool = False, dataset_name: str = "globalscale", ) -> None:
    """Precompute GLG for Globalscale tiles [start_id, end_id) and persist complex state.

    Layout per tile:
      output_dir/tile_{id}/glg_state.pkl  # includes igraph, numpy arrays, sets, metadata
      output_dir/tile_{id}/rtree.{dat,idx}  # disk-backed spatial index
    """
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == 'globalscale':
        gt_graph_pattern = os.path.join(input_dir, 'region_{}_refine_gt_graph.p')
        config = build_default_config_globalscale()
    elif dataset_name == 'wildroad':
        gt_graph_pattern = os.path.join(input_dir, 'gt_graph_{}.pickle')
        config = build_default_config_wild_road()
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')

    coord_transform = lambda v: v[:, ::-1]

    start_time = time.time()

    total = 0
    for tile_id in range(int(start_id), int(end_id)):
        tile_dir = os.path.join(output_dir, f'tile_{tile_id}')
        state_path = os.path.join(tile_dir, 'glg_state.pkl')
        if os.path.exists(state_path) and not overwrite:
            print(f'[skip] {tile_dir} already exists')
            continue

        gt_path = gt_graph_pattern.format(tile_id)
        if not os.path.exists(gt_path):
            print(f'[warn] GT not found for tile {tile_id}: {gt_path}')
            continue

        try:
            with open(gt_path, 'rb') as f:
                gt_graph_adj = pickle.load(f)
            if len(gt_graph_adj) == 0:
                print(f'[skip] empty tile {tile_id}')
                continue

            t0 = time.time()
            glg = GraphLabelGenerator(config, gt_graph_adj, coord_transform)
            glg.save_precomputed(tile_dir)
            dt = time.time() - t0
            total += 1
            print(f'[ok] saved {tile_dir} in {dt:.2f}s')
        except Exception as e:
            print(f'[error] tile {tile_id}: {e}')

    end_time = time.time()
    print(f'Time taken: {end_time - start_time:.2f} seconds')
    print(f'Done. Saved {total} tiles to {output_dir}.')


def main():
    parser = argparse.ArgumentParser(
        description='Precompute and persist GLG (igraph + Rtree) for Globalscale and Wild Road tiles.'
    )
    parser.add_argument('--dataset_name', type=str, default="wildroad", help='dataset name')
    parser.add_argument('--start_id', type=int, required=True, help='inclusive start id')
    parser.add_argument('--end_id', type=int, required=True, help='exclusive end id')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./wildroad_GLG/test_patches/test_AB_GLG',
        help='directory to write tile_{id}/ with glg_state.pkl and rtree files',
    )
    parser.add_argument('--input_dir', type=str, required=True, help='directory containing gt_graph_{id}.pickle files')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing files')
    args = parser.parse_args()

    precompute_and_save_range(args.start_id, args.end_id, args.output_dir, args.input_dir, overwrite=args.overwrite, dataset_name=args.dataset_name)


if __name__ == '__main__':
    main()

