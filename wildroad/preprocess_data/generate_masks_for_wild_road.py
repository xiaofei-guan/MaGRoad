import os
import argparse
import glob
import shutil
import pickle
import numpy as np
import networkx as nx
import cv2

# Constants for image generation
IMAGE_SIZE = 1024
KEYPOINT_RADIUS = 10
ROAD_WIDTH = 10

def create_directory(dir_path, delete=False):
    """Create directory. If delete=True, remove it first if it exists."""
    if os.path.isdir(dir_path) and delete:
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def draw_points_on_image(size, points, radius):
    """Draws filled circles for points on a square image."""
    image = np.zeros((size, size), dtype=np.uint8)
    for point in points:
        cv2.circle(image, point, radius, 255, -1)
    return image

def draw_line_segments_on_image(size, line_segments, width):
    """Draws lines for segments on a square image."""
    image = np.zeros((size, size), dtype=np.uint8)
    for segment in line_segments:
        (x1, y1), (x2, y2) = segment
        cv2.line(image, (x1, y1), (x2, y2), 255, width)
    return image

def process_tile(file_path, output_subdir, tile_index):
    """Reads graph pickle, generates masks, and saves to output directory."""
    try:
        with open(file_path, 'rb') as f:
            gt_graph = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load {file_path}: {e}")
        return

    # Build NetworkX graph from adjacency dictionary
    graph = nx.Graph()
    for n, neis in gt_graph.items():
        for nei in neis:
            # Convert coordinates (y, x) -> (x, y) if needed based on original logic
            graph.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))
    
    # Identify key nodes (intersections/endpoints, degree != 2)
    key_nodes = []
    for node, degree in graph.degree():
        if degree != 2:
            key_nodes.append(node)

    # Generate masks
    keypoint_mask = draw_points_on_image(size=IMAGE_SIZE, points=key_nodes, radius=KEYPOINT_RADIUS)
    road_mask = draw_line_segments_on_image(size=IMAGE_SIZE, line_segments=graph.edges(), width=ROAD_WIDTH)

    # Save outputs
    cv2.imwrite(os.path.join(output_subdir, f'keypoint_mask_{tile_index}.png'), keypoint_mask)
    cv2.imwrite(os.path.join(output_subdir, f'road_mask_{tile_index}.png'), road_mask)

def main():
    parser = argparse.ArgumentParser(description="Generate road and keypoint masks from graph pickle files.")
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory containing *_patches folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Root directory to save generated masks')
    args = parser.parse_args()

    # The dataset structure typically contains these folders
    patches_dirs = ['train_patches', 'val_patches', 'test_patches']

    for p_dir in patches_dirs:
        # Construct path: input_dir/train_patches
        current_input_path = os.path.join(args.input_dir, p_dir)
        
        if not os.path.exists(current_input_path):
            print(f"[WARN] Directory not found: {current_input_path}")
            continue

        # Find subdirectories ending with _AB (e.g., train_AB)
        if os.path.isdir(current_input_path):
            subdirs = [d for d in os.listdir(current_input_path) if os.path.isdir(os.path.join(current_input_path, d))]
            # Filter for folders ending in _AB as requested
            target_subdirs = [d for d in subdirs if d.endswith('_AB')]
        else:
            target_subdirs = []

        for sub in target_subdirs:
            input_subdir_full = os.path.join(current_input_path, sub)
            # Replicate structure in output
            output_subdir_full = os.path.join(args.output_dir, p_dir, sub)

            # Find all pickle files to process (gt_graph_*.pickle)
            pickle_pattern = os.path.join(input_subdir_full, 'gt_graph_*.pickle')
            pickle_files = glob.glob(pickle_pattern)
            
            if not pickle_files:
                print(f"[WARN] No pickle files found in {input_subdir_full}")
                continue

            print(f"[INFO] Found {len(pickle_files)} files in {sub}. Generating masks...")
            create_directory(output_subdir_full, delete=False)

            count = 0
            for pkl_path in pickle_files:
                filename = os.path.basename(pkl_path)
                # Extract index from filename: gt_graph_{index}.pickle
                try:
                    # Remove prefix and suffix to get index
                    index_part = filename.replace('gt_graph_', '').replace('.pickle', '')
                    tile_index = int(index_part)
                except ValueError:
                    print(f"[SKIP] Invalid filename format: {filename}")
                    continue
                
                process_tile(pkl_path, output_subdir_full, tile_index)
                count += 1
                if count % 500 == 0:
                    print(f"  Processed {count}...")
            
            print(f"[DONE] Finished {sub}. Total: {count}")

if __name__ == "__main__":
    main()
