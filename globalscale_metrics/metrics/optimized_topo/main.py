import argparse
import math
import os
import pickle
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-graph_gt', required=True, type=str, help='ground truth graph (pickle dict x->neighbors)')
    parser.add_argument('-graph_prop', required=True, type=str, help='proposed graph (pickle dict x->neighbors)')
    parser.add_argument('-matching_threshold', type=float, default=0.00010, help='topo marble-hole matching distance')
    parser.add_argument('-interval', dest='topo_interval', type=float, default=0.00005, help='topo marble-hole interval')
    parser.add_argument('-lat_top_left', type=float, default=41.0)
    parser.add_argument('-lon_top_left', type=float, default=-71.0)
    parser.add_argument('-r', type=float, default=None, help='propagation distance override')
    parser.add_argument('-output', type=str, default='', help="output txt file path; if empty, only print")
    parser.add_argument('-workers', type=int, default=None, help='number of worker threads; set 1 on Linux if hangs')
    args = parser.parse_args()

    # local imports
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if cur_dir not in sys.path:
        sys.path.insert(0, cur_dir)
    import topo as topo
    from graph import RoadGraph

    lat_top_left = args.lat_top_left
    lon_top_left = args.lon_top_left

    map1 = pickle.load(open(args.graph_gt, 'rb'))
    map2 = pickle.load(open(args.graph_prop, 'rb'))

    min_lat = lat_top_left
    max_lon = lon_top_left

    def xy2latlon(x, y):
        lat = lat_top_left - x * 1.0 / 111111.0
        lon = lon_top_left + (y * 1.0 / 111111.0) / math.cos(math.radians(lat_top_left))
        return lat, lon

    def create_graph(m):
        nonlocal min_lat, max_lon
        graph = RoadGraph()
        nid = 0
        idmap = {}
        for k, v in m.items():
            n1 = k
            lat1, lon1 = xy2latlon(n1[0], n1[1])
            if lat1 < min_lat:
                min_lat = lat1
            if lon1 > max_lon:
                max_lon = lon1
            for n2 in v:
                lat2, lon2 = xy2latlon(n2[0], n2[1])
                if n1 in idmap:
                    id1 = idmap[n1]
                else:
                    id1 = nid
                    idmap[n1] = nid
                    nid += 1
                if n2 in idmap:
                    id2 = idmap[n2]
                else:
                    id2 = nid
                    idmap[n2] = nid
                    nid += 1
                graph.addEdge(id1, lat1, lon1, id2, lat2, lon2)
        graph.ReverseDirectionLink()
        # give positive scores to include all edges/nodes (compat)
        for node in list(graph.nodes.keys()):
            graph.nodeScore[node] = 100
        for edge in list(graph.edges.keys()):
            graph.edgeScore[edge] = 100
        return graph

    graph_gt = create_graph(map1)
    graph_prop = create_graph(map2)

    # region padding mimics original topo/main.py
    region = [
        min_lat - 300 * 1.0 / 111111.0,
        lon_top_left - 500 * 1.0 / 111111.0,
        lat_top_left + 300 * 1.0 / 111111.0,
        max_lon + 500 * 1.0 / 111111.0,
    ]
    graph_gt.region = region
    graph_prop.region = region

    # starting points on GT
    losm = topo.TOPOGenerateStartingPoints(graph_gt, region=region, image='NULL', check=False, direction=False, metaData=None)
    # pair GPS (prop) edges with OSM (gt) starts
    lmap = topo.TOPOGeneratePairs(graph_prop, graph_gt, losm, threshold=args.matching_threshold, region=region)

    # propagation distance per dataset size
    if args.r is not None:
        r = float(args.r)
    else:
        r = 0.00300
        if lat_top_left - min_lat < 0.01000:
            r = 0.00150

    # Determine worker count (CLI > env > default)
    env_workers = os.environ.get("TOPO_MAX_WORKERS", "")
    try:
        env_workers_val = int(env_workers) if env_workers.strip() != "" else None
    except ValueError:
        env_workers_val = None
    worker_count = args.workers if args.workers is not None else (env_workers_val if env_workers_val is not None else (os.cpu_count() or 4))

    p, overall_recall = topo.TOPOWithPairs(
        graph_prop,
        graph_gt,
        lmap,
        losm,
        r=r,
        step=args.topo_interval,
        threshold=args.matching_threshold,
        one2oneMatching=True,
        metaData=None,
        max_workers=worker_count,
    )

    line = f"precision={p} overall-recall={overall_recall}"
    # Print always
    print(line)
    # Optionally also write to file if provided
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(args.output, 'w') as f:
            f.write(line)


if __name__ == '__main__':
    main()


