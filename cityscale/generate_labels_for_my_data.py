import json
import os
import numpy as np
import shutil
import pickle
import networkx as nx
import cv2
import ast

# Modified to support rectangular images
IMAGE_WIDTH = 11904
IMAGE_HEIGHT = 12064
KEYPOINT_RADIUS = 10
ROAD_WIDTH = 8
output_dir = '/home/guanwenfei/ResearchWork/sam_road-main/mydata/2023contest/'


def create_directory(dir, delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


def draw_points_on_image(width, height, points, radius):
    """
    Draw points on a rectangular image

    Parameters:
    - width: Image width (pixels)
    - height: Image height (pixels)
    - points: List of point coordinates, each as (x, y)
    - radius: Radius of each point (pixels)

    Returns:
    - Rectangular image with drawn points
    """
    # Create a blank image of specified size
    image = np.zeros((height, width), dtype=np.uint8)

    # Draw each point on the image
    for point in points:
        cv2.circle(image, point, radius, 255, -1)

    return image


def draw_line_segments_on_image(width, height, line_segments, line_width):
    """
    Draw line segments on a rectangular image

    Parameters:
    - width: Image width (pixels)
    - height: Image height (pixels)
    - line_segments: List of line segments, each as ((x1, y1), (x2, y2))
    - line_width: Line width (pixels)

    Returns:
    - Rectangular image with drawn line segments
    """
    # Create a blank image of specified size
    image = np.zeros((height, width), dtype=np.uint8)

    # Draw each line segment on the image
    for segment in line_segments:
        (x1, y1), (x2, y2) = segment
        cv2.line(image, (x1, y1), (x2, y2), 255, line_width)

    return image


def load_road_network(json_path):
    """Load road network from JSON file and parse coordinate strings to tuples"""
    with open(json_path, 'r') as f:
        road_network = json.load(f)

    # Convert string coordinates to actual tuples
    parsed_network = {}
    for node, neighbors in road_network.items():
        node_tuple = ast.literal_eval(node)
        neighbor_tuples = [ast.literal_eval(n) for n in neighbors]
        parsed_network[node_tuple] = neighbor_tuples

    return parsed_network


create_directory(output_dir, delete=False)

for tile_index in range(1):
    print(f'Processing 2023contest tile {tile_index}.')
    vertices = []
    edges = []
    vertex_flag = True

    saved_data = {}
    json_path = "/home/guanwenfei/ResearchWork/sam_road-main/mydata/2023contest/road_network_processed/adjacency.json"

    # Load road network
    gt_graph = load_road_network(json_path)

    # Convert adjacency list coordinates from XY to YX
    yx_adjacency = {}
    for node, neighbors in gt_graph.items():
        yx_node = (node[1], node[0])  # Swap Y and X
        yx_neighbors = [(n[1], n[0]) for n in neighbors]  # Swap Y and X for neighbors
        yx_adjacency[yx_node] = yx_neighbors

    # Save the YX adjacency list as a pickle file
    pickle_path = "/home/guanwenfei/ResearchWork/sam_road-main/mydata/2023contest/road_network_processed/adjacency.p"
    with open(pickle_path, 'wb') as f:
        pickle.dump(yx_adjacency, f)

    graph = nx.Graph()  # Undirected graph
    for n, neis in gt_graph.items():
        for nei in neis:
            # Coordinates in JSON file are (x, y)
            graph.add_edge((int(n[0]), int(n[1])), (int(nei[0]), int(nei[1])))

    # Collect key nodes (nodes with degree not equal to 2)
    key_nodes = []
    for node, degree in graph.degree():
        if degree != 2:
            key_nodes.append(node)

    saved_data['ke_coors'] = key_nodes
    # Save data to JSON file
    json_file_path = f'./keypoints/region_{tile_index}.json'
    with open(json_file_path, 'w') as f:
        json.dump(saved_data, f, indent=4)

    print(len(key_nodes))

    # Create keypoint mask
    keypoint_mask = draw_points_on_image(
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
        points=key_nodes, radius=KEYPOINT_RADIUS
    )

    # Create road mask
    road_mask = draw_line_segments_on_image(
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
        line_segments=graph.edges(), line_width=ROAD_WIDTH
    )

    cv2.imwrite(os.path.join(output_dir, f'keypoint_mask_{tile_index}.png'), keypoint_mask)
    cv2.imwrite(os.path.join(output_dir, f'road_mask_{tile_index}.png'), road_mask)