import os
import numpy as np 
import shutil
import pickle
import networkx as nx
import cv2
import argparse
import glob
import re

IMAGE_SIZE = 2048
KEYPOINT_RADIUS = 3
ROAD_WIDTH = 3

def create_directory(dir, delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def draw_points_on_image(size, points, radius):
    """
    Draws points on a square image using OpenCV.
    """
    # Create a square image of the given size, initialized to zeros (black), with one channel (grayscale), and dtype uint8
    image = np.zeros((size, size), dtype=np.uint8)

    # Iterate through the list of points
    for point in points:
        cv2.circle(image, point, radius, 255, -1)

    return image

def draw_line_segments_on_image(size, line_segments, width):
    """
    Draws line segments on a square image using OpenCV.
    """
    # Create a square image of the given size, initialized to zeros (black)
    # with one channel (grayscale), and dtype uint8
    image = np.zeros((size, size), dtype=np.uint8)

    # Iterate through the list of line segments
    for segment in line_segments:
        (x1, y1), (x2, y2) = segment
        cv2.line(image, (x1, y1), (x2, y2), 255, width)

    return image

def process_subset(input_root, output_root, subset_name):
    input_path = os.path.join(input_root, subset_name)
    output_path = os.path.join(output_root, subset_name)
    
    if not os.path.exists(input_path):
        print(f"[WARN] Input directory not found: {input_path}. Skipping {subset_name}.")
        return

    # Create output directory for this subset
    create_directory(output_path, delete=False)
    
    # Find all graph files
    # Pattern: region_{id}_refine_gt_graph.p
    pattern = os.path.join(input_path, "region_*_refine_gt_graph.p")
    files = glob.glob(pattern)
    
    if not files:
        print(f"[WARN] No graph files found in {input_path}")
        return
        
    print(f"[INFO] Processing {subset_name}: found {len(files)} files.")
    
    count = 0
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            # Extract tile index from filename
            match = re.search(r'region_(\d+)_refine_gt_graph\.p', filename)
            if not match:
                continue
                
            tile_index = int(match.group(1))
            
            # Load GT Graph
            with open(file_path, 'rb') as f:
                gt_graph = pickle.load(f)
                
            graph = nx.Graph()  # undirected
            for n, neis in gt_graph.items():
                for nei in neis:
                    graph.add_edge((int(n[1]), int(n[0])), (int(nei[1]), int(nei[0])))
            
            # Collect key nodes (degree != 2)
            key_nodes = []
            for node, degree in graph.degree():
                if degree != 2:
                    key_nodes.append(node)

            # Create key point mask
            keypoint_mask = draw_points_on_image(size=IMAGE_SIZE, points=key_nodes, radius=KEYPOINT_RADIUS)

            # Create road mask
            road_mask = draw_line_segments_on_image(
                size=IMAGE_SIZE, line_segments=graph.edges(), width=ROAD_WIDTH)

            cv2.imwrite(os.path.join(output_path, f'keypoint_mask_{tile_index}.png'), keypoint_mask)
            cv2.imwrite(os.path.join(output_path, f'road_mask_{tile_index}.png'), road_mask)
            
            count += 1
            if count % 100 == 0:
                print(f"  Processed {count}/{len(files)}...")
                
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

    print(f"[DONE] Finished {subset_name}, generated {count} mask pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masks for GlobalScale dataset (train/out_of_domain)")
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory containing train and out_of_domain folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Root output directory for masks')
    
    args = parser.parse_args()
    
    subsets = ['train', 'out_of_domain']
    
    print(f"Generating masks from {args.input_dir} to {args.output_dir}")
    
    for subset in subsets:
        process_subset(args.input_dir, args.output_dir, subset)
