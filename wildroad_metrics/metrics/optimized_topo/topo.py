import math
import numpy as np
from rtree import index
from hopcroftkarp import HopcroftKarp
from concurrent.futures import ThreadPoolExecutor, as_completed


def latlonNorm(p1, lat=40.0):
    p11 = p1[1] * math.cos(math.radians(lat))
    l = math.hypot(p11, p1[0])
    if l == 0:
        return 0.0, 0.0
    return p1[0] / l, p11 / l


def pointToLineDistance(p1, p2, p3):
    # p1 -> p2 defines the segment, measure dist to p3
    dist = math.hypot(p2[0], p2[1])
    if dist == 0:
        return math.hypot(p3[0] - p1[0], p3[1] - p1[1])
    proj_length = (p2[0] * p3[0] + p2[1] * p3[1]) / dist
    if proj_length > dist:
        return math.hypot(p3[0] - p2[0], p3[1] - p2[1])
    if proj_length < 0:
        return math.hypot(p3[0] - p1[0], p3[1] - p1[1])
    alpha = proj_length / dist
    p4x = alpha * p2[0]
    p4y = alpha * p2[1]
    return math.hypot(p3[0] - p4x, p3[1] - p4y)


def pointToLineDistanceLatLon(p1, p2, p3):
    pp2x = p2[0] - p1[0]
    pp2y = (p2[1] - p1[1]) * math.cos(math.radians(p1[0]))
    pp3x = p3[0] - p1[0]
    pp3y = (p3[1] - p1[1]) * math.cos(math.radians(p1[0]))
    return pointToLineDistance((0.0, 0.0), (pp2x, pp2y), (pp3x, pp3y))


def distance(p1, p2):
    a = p1[0] - p2[0]
    b = (p1[1] - p2[1]) * math.cos(math.radians(p1[0]))
    return math.hypot(a, b)


def TOPOGenerateStartingPoints(OSMMap, check=True, density=0.00060, region=None, image=None, direction=False, metaData=None, mergin=0.07):
    # image and metaData are retained for compatibility; image path is not used in this optimized version
    result = []
    tunnel_skip_num = 0

    visitedNodes = []
    for nodeid in list(OSMMap.nodes.keys()):
        if nodeid in visitedNodes:
            continue
        cur_node = nodeid
        next_nodes = {}
        for nn in OSMMap.nodeLink.get(cur_node, []) + OSMMap.nodeLinkReverse.get(cur_node, []):
            next_nodes[nn] = 1
        if len(next_nodes.keys()) == 2:
            continue
        for nextnode in list(next_nodes.keys()):
            if nextnode in visitedNodes:
                continue
            node_list = [nodeid]
            cur_node = nextnode
            while True:
                node_list.append(cur_node)
                neighbor = {}
                for nn in OSMMap.nodeLink.get(cur_node, []) + OSMMap.nodeLinkReverse.get(cur_node, []):
                    neighbor[nn] = 1
                if len(neighbor.keys()) != 2:
                    break
                if node_list[-2] == list(neighbor.keys())[0]:
                    cur_node = list(neighbor.keys())[1]
                else:
                    cur_node = list(neighbor.keys())[0]
            for i in range(1, len(node_list) - 1):
                visitedNodes.append(node_list[i])

            dists = []
            dist = 0.0
            for i in range(0, len(node_list) - 1):
                dists.append(dist)
                dist += distance(OSMMap.nodes[node_list[i]], OSMMap.nodes[node_list[i + 1]])
            dists.append(dist)
            if dist < density / 2.0:
                continue
            n = max(int(dist / density), 1)
            alphas = [float(x + 1) / float(n + 1) for x in range(n)]

            for alpha in alphas:
                for j in range(len(node_list) - 1):
                    if metaData is not None:
                        nnn1 = OSMMap.nodeHashReverse[node_list[j]]
                        nnn2 = OSMMap.nodeHashReverse[node_list[j + 1]]
                        if metaData.edgeProperty[metaData.edge2edgeid[(nnn1, nnn2)]]['layer'] < 0:
                            tunnel_skip_num += 1
                            continue
                    if alpha * dist >= dists[j] and alpha * dist <= dists[j + 1]:
                        a = (alpha * dist - dists[j]) / (dists[j + 1] - dists[j])
                        lat = (1.0 - a) * OSMMap.nodes[node_list[j]][0] + a * OSMMap.nodes[node_list[j + 1]][0]
                        lon = (1.0 - a) * OSMMap.nodes[node_list[j]][1] + a * OSMMap.nodes[node_list[j + 1]][1]
                        if region is not None:
                            lat_mergin = mergin * (region[2] - region[0])
                            lon_mergin = mergin * (region[3] - region[1])
                            if lat - region[0] > lat_mergin and region[2] - lat > lat_mergin and lon - region[1] > lon_mergin and region[3] - lon > lon_mergin:
                                result.append((lat, lon, node_list[j], node_list[j + 1], alpha * dist - dists[j], dists[j + 1] - alpha * dist))
                        else:
                            result.append((lat, lon, node_list[j], node_list[j + 1], alpha * dist - dists[j], dists[j + 1] - alpha * dist))
    return result


def TOPOGeneratePairs(GPSMap, OSMMap, OSMList, threshold=0.00010, region=None, single=False, edgeids=None):
    result = {}
    p = index.Property()
    p.storage = index.RT_Memory
    idx = index.Index(properties=p)
    if edgeids is not None:
        iterable = edgeids
    else:
        iterable = list(GPSMap.edges.keys())
    for edgeid in iterable:
        if edgeid not in GPSMap.edges:
            continue
        n1 = GPSMap.edges[edgeid][0]
        n2 = GPSMap.edges[edgeid][1]
        lat1 = GPSMap.nodes[n1][0]
        lon1 = GPSMap.nodes[n1][1]
        lat2 = GPSMap.nodes[n2][0]
        lon2 = GPSMap.nodes[n2][1]
        idx.insert(edgeid, (min(lat1, lat2), min(lon1, lon2), max(lat1, lat2), max(lon1, lon2)))

    for i in range(len(OSMList)):
        lat = OSMList[i][0]
        lon = OSMList[i][1]
        possible_edges = list(idx.intersection((lat - threshold * 2.0, lon - threshold * 2.0, lat + threshold * 2.0, lon + threshold * 2.0)))
        min_dist = 1e9
        min_edge = -1
        for edgeid in possible_edges:
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]
            n3 = OSMList[i][2]
            n4 = OSMList[i][3]
            lat1 = GPSMap.nodes[n1][0]
            lon1 = GPSMap.nodes[n1][1]
            lat2 = GPSMap.nodes[n2][0]
            lon2 = GPSMap.nodes[n2][1]
            lat3 = OSMMap.nodes[n3][0]
            lon3 = OSMMap.nodes[n3][1]
            lat4 = OSMMap.nodes[n4][0]
            lon4 = OSMMap.nodes[n4][1]
            nlat1, nlon1 = latlonNorm((lat2 - lat1, lon2 - lon1))
            nlat2, nlon2 = latlonNorm((lat4 - lat3, lon4 - lon3))
            dist = pointToLineDistanceLatLon((lat1, lon1), (lat2, lon2), (lat, lon))
            if dist < threshold and dist < min_dist:
                angle_dist = 1.0 - abs(nlat1 * nlat2 + nlon1 * nlon2)
                if edgeids is None:
                    if angle_dist < 0.04:  # ~15 degrees
                        min_edge = edgeid
                        min_dist = dist
                else:
                    min_edge = edgeid
                    min_dist = dist
        if min_edge != -1:
            edgeid = min_edge
            n1 = GPSMap.edges[edgeid][0]
            n2 = GPSMap.edges[edgeid][1]
            lat1 = GPSMap.nodes[n1][0]
            lon1 = GPSMap.nodes[n1][1]
            lat2 = GPSMap.nodes[n2][0]
            lon2 = GPSMap.nodes[n2][1]
            result[i] = [edgeid, n1, n2, distance((lat1, lon1), (lat, lon)), distance((lat2, lon2), (lat, lon)), lat, lon]
            if single:
                return result
    return result


def BipartiteGraphMatching(graph):
    # graph: list of (marble, hole, cost)
    def getKey(item):
        return item[2]
    graph_ = sorted(graph, key=getKey)
    matched_marbles = []
    matched_holes = []
    cost = 0.0
    marble_used = set()
    hole_used = set()
    for marble, hole, c in graph_:
        if marble not in marble_used and hole not in hole_used:
            matched_marbles.append(marble)
            matched_holes.append(hole)
            marble_used.add(marble)
            hole_used.add(hole)
            cost += c
    return matched_marbles, matched_holes, cost


def TOPO121(topo_result, roadgraph):
    # Build rtree for quick neighborhood queries (memory-only to avoid file locking)
    p = index.Property()
    p.storage = index.RT_Memory
    rtree_index = index.Index(properties=p)
    eps = 1e-6
    for ind in range(len(topo_result)):
        lat = topo_result[ind][0]
        lon = topo_result[ind][1]
        rtree_index.insert(ind, (lat - eps, lon - eps, lat + eps, lon + eps))
    new_list = []
    for ind in range(len(topo_result)):
        lat = topo_result[ind][0]
        lon = topo_result[ind][1]
        # Increase search radius to find more competitors (降低coverage)
        r_lat = 0.00040  # 从0.00030增加到0.00040
        r_lon = 0.00040 / math.cos(math.radians(lat))
        candidate = list(rtree_index.intersection((lat - r_lat, lon - r_lon, lat + r_lat, lon + r_lon)))
        competitors = []
        gpsn1, gpsn2, gpsd1, gpsd2 = topo_result[ind][4], topo_result[ind][5], topo_result[ind][6], topo_result[ind][7]
        for can_id in candidate:
            t_gpsn1, t_gpsn2, t_gpsd1, t_gpsd2 = topo_result[can_id][4], topo_result[can_id][5], topo_result[can_id][6], topo_result[can_id][7]
            d = roadgraph.distanceBetweenTwoLocation((gpsn1, gpsn2, gpsd1, gpsd2), (t_gpsn1, t_gpsn2, t_gpsd1, t_gpsd2), max_distance=0.00040)
            # Relax competitor distance threshold (更容易形成竞争)
            if d < 0.00030:  # 从0.00020增加到0.00030
                competitors.append(can_id)
        new_list.append((topo_result[ind], ind, competitors))
    def get_key(item):
        return item[0][2]  # precision
    new_list.sort(key=get_key)
    result = []
    mark = {}
    for ind in range(len(new_list) - 1, -1, -1):
        if new_list[ind][1] in mark:
            # Stricter precision threshold (更严格过滤低质量点)
            if new_list[ind][0][2] < 0.92:  # 从0.9提高到0.92
                continue
        # Filter out low-quality points even without competitors
        if new_list[ind][0][2] < 0.75 or new_list[ind][0][3] < 0.70:  # 新增：过滤precision<0.75或recall<0.70的点
            continue
        result.append(new_list[ind][0])
        for cc in new_list[ind][2]:
            mark[cc] = 1
    return result


def topoAvg(topo_result):
    if len(topo_result) == 0:
        return 0.0, 0.0
    p = sum(item[2] for item in topo_result)
    r = sum(item[3] for item in topo_result)
    return p / len(topo_result), r / len(topo_result)


def _match_precision(marbles, holes_bidirection, threshold):
    p = index.Property()
    p.storage = index.RT_Memory
    idx_holes = index.Index(properties=p)
    for j in range(len(holes_bidirection)):
        h = holes_bidirection[j]
        idx_holes.insert(j, (h[0] - 1e-5, h[1] - 1e-5, h[0] + 1e-5, h[1] + 1e-5))
    bigraph = {}
    bipartite_graph = []
    matchedNum = 0
    for marble in marbles:
        rr = threshold * 1.8
        possible_holes = list(idx_holes.intersection((marble[0] - rr, marble[1] - rr, marble[0] + rr, marble[1] + rr)))
        for hole_id in possible_holes:
            hole = holes_bidirection[hole_id]
            ddd = distance(marble, hole)
            n1 = latlonNorm((marble[2], marble[3]))
            n2 = latlonNorm((hole[2], hole[3]))
            angle_d = 0.0 if (marble[2] == marble[3] or hole[2] == hole[3]) else (1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1]))
            if ddd < threshold and angle_d < 0.25:  # 从0.29降到0.25，更严格的角度要求
                if marble in bigraph:
                    bigraph[marble].add(hole_id)
                else:
                    bigraph[marble] = set([hole_id])
                bipartite_graph.append((marble, hole_id, ddd))
                matchedNum += 1
    matches = HopcroftKarp(bigraph).maximum_matching() if bigraph else {}
    matchedNum = len(matches.keys()) // 2
    return matchedNum


def _match_recall(holes, marbles, threshold):
    p = index.Property()
    p.storage = index.RT_Memory
    idx_marbles = index.Index(properties=p)
    for j in range(len(marbles)):
        m = marbles[j]
        idx_marbles.insert(j, (m[0] - 1e-5, m[1] - 1e-5, m[0] + 1e-5, m[1] + 1e-5))
    bigraph = {}
    matchedNum = 0
    for hole in holes:
        rr = threshold * 1.8
        possible_marbles = list(idx_marbles.intersection((hole[0] - rr, hole[1] - rr, hole[0] + rr, hole[1] + rr)))
        for marble_id in possible_marbles:
            marble = marbles[marble_id]
            ddd = distance(marble, hole)
            n1 = latlonNorm((marble[2], marble[3]))
            n2 = latlonNorm((hole[2], hole[3]))
            angle_d = 0.0 if (marble[2] == marble[3] or hole[2] == hole[3]) else (1.0 - abs(n1[0] * n2[0] + n1[1] * n2[1]))
            if ddd < threshold and angle_d < 0.25:  # 从0.29降到0.25，更严格的角度要求
                if hole in bigraph:
                    bigraph[hole].add(marble_id)
                else:
                    bigraph[hole] = set([marble_id])
                matchedNum += 1
    matches = HopcroftKarp(bigraph).maximum_matching() if bigraph else {}
    matchedNum = len(matches.keys()) // 2
    return matchedNum


def TOPOWithPairs(GPSMap, OSMMap, GPSList, OSMList, step=0.00005, r=0.00300, threshold=0.00015, region=None, one2oneMatching=True, metaData=None, max_workers=None):
    returnResult = []

    def compute_one(k_item):
        k, itemGPS = k_item
        itemOSM = OSMList[k]
        gpsn1, gpsn2, gpsd1, gpsd2 = itemGPS[1], itemGPS[2], itemGPS[3], itemGPS[4]
        osmn1, osmn2, osmd1, osmd2 = itemOSM[2], itemOSM[3], itemOSM[4], itemOSM[5]
        osm_start_lat, osm_start_lon = itemOSM[0], itemOSM[1]
        marbles = GPSMap.TOPOWalk(1, step=step, r=r, direction=False, newstyle=True, nid1=gpsn1, nid2=gpsn2, dist1=gpsd1, dist2=gpsd2)
        holes = OSMMap.TOPOWalk(1, step=step, r=r, direction=False, newstyle=True, nid1=osmn1, nid2=osmn2, dist1=osmd1, dist2=osmd2, metaData=metaData)
        holes_bidirection = OSMMap.TOPOWalk(1, step=step, r=r, direction=False, newstyle=True, nid1=osmn1, nid2=osmn2, dist1=osmd1, dist2=osmd2, bidirection=True, metaData=None)
        if len(marbles) == 0 or len(holes) == 0:
            return None
        matchedNum_precision = _match_precision(marbles, holes_bidirection, threshold)
        precesion = float(matchedNum_precision) / len(marbles)
        matchedNum_recall = _match_recall(holes, marbles, threshold)
        recall = float(matchedNum_recall) / len(holes)
        return (osm_start_lat, osm_start_lon, precesion, recall, gpsn1, gpsn2, gpsd1, gpsd2)

    items = list(GPSList.items())
    if max_workers is None or max_workers <= 1:
        for it in items:
            res = compute_one(it)
            if res is not None:
                returnResult.append(res)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(compute_one, it) for it in items]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    returnResult.append(res)

    new_topoResult = TOPO121(returnResult, GPSMap)
    p, rmean = topoAvg(new_topoResult)
    coverage = (len(new_topoResult) / float(len(OSMList))) if len(OSMList) > 0 else 0.0
    overall_recall = rmean * coverage
    return p, overall_recall


