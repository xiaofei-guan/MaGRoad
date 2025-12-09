import math
import numpy as np


def distance(p1, p2):
    a = p1[0] - p2[0]
    b = (p1[1] - p2[1]) * math.cos(math.radians(p1[0]))
    return math.hypot(a, b)


class RoadGraph:
    def __init__(self):
        self.nodeHash = {}  # original external id -> local id
        self.nodeHashReverse = {}  # local id -> external id
        self.nodes = {}  # local id -> [lat, lon]
        self.edges = {}  # edge id -> [local_n1, local_n2]
        self.nodeLink = {}  # local id -> [neighbors]
        self.nodeLinkReverse = {}  # local id -> [reverse neighbors]
        self.nodeScore = {}
        self.edgeScore = {}
        self.edgeHash = {}  # (local_n1 * 1e7 + local_n2) -> edge id
        self.nodeID = 0
        self.edgeID = 0
        self.region = None

    def addEdge(self, nid1, lat1, lon1, nid2, lat2, lon2, edgeScore=0):
        if nid1 not in self.nodeHash:
            self.nodeHash[nid1] = self.nodeID
            self.nodeHashReverse[self.nodeID] = nid1
            self.nodes[self.nodeID] = [lat1, lon1]
            self.nodeLink[self.nodeID] = []
            self.nodeLinkReverse[self.nodeID] = []
            self.nodeScore[self.nodeID] = 0
            self.nodeID += 1

        if nid2 not in self.nodeHash:
            self.nodeHash[nid2] = self.nodeID
            self.nodeHashReverse[self.nodeID] = nid2
            self.nodes[self.nodeID] = [lat2, lon2]
            self.nodeLink[self.nodeID] = []
            self.nodeLinkReverse[self.nodeID] = []
            self.nodeScore[self.nodeID] = 0
            self.nodeID += 1

        localid1 = self.nodeHash[nid1]
        localid2 = self.nodeHash[nid2]

        h = localid1 * 10000000 + localid2
        if h in self.edgeHash:
            return

        self.edges[self.edgeID] = [localid1, localid2]
        self.edgeHash[h] = self.edgeID
        self.edgeScore[self.edgeID] = edgeScore
        self.edgeID += 1

        if localid2 not in self.nodeLink[localid1]:
            self.nodeLink[localid1].append(localid2)
        if localid1 not in self.nodeLinkReverse[localid2]:
            self.nodeLinkReverse[localid2].append(localid1)

    def ReverseDirectionLink(self):
        # ensure reverse adjacency complete
        for edge in list(self.edges.values()):
            n1, n2 = edge[0], edge[1]
            if n1 not in self.nodeLinkReverse:
                self.nodeLinkReverse[n1] = []
            if n2 not in self.nodeLinkReverse:
                self.nodeLinkReverse[n2] = []
            if n1 not in self.nodeLinkReverse[n2]:
                self.nodeLinkReverse[n2].append(n1)
        for nodeId in list(self.nodes.keys()):
            if nodeId not in self.nodeLinkReverse:
                self.nodeLinkReverse[nodeId] = []

    def _vectorized_samples_on_segment(self, lat1, lon1, lat2, lon2, start_cur, end_cur, step):
        # returns arrays of (lat, lon) sampled along [start_cur, end_cur) with spacing step
        if end_cur <= start_cur:
            return None, None
        num = int(max(0, math.floor((end_cur - start_cur) / step)))
        if num <= 0:
            return None, None
        # positions along the segment in meters-equivalent along curve parameterized by arc length proxy
        cur_vals = start_cur + step * np.arange(num, dtype=np.float64)
        if cur_vals.size == 0:
            return None, None
        l = distance((lat1, lon1), (lat2, lon2))
        if l <= 0:
            return None, None
        alpha = (cur_vals / l).astype(np.float64)
        alpha = np.clip(alpha, 0.0, 1.0)
        lat = lat2 * alpha + lat1 * (1.0 - alpha)
        lon = lon2 * alpha + lon1 * (1.0 - alpha)
        return lat, lon

    def TOPOWalk(self, nodeid, step=0.00005, r=0.00300, direction=False,
                 newstyle=False, nid1=0, nid2=0, dist1=0.0, dist2=0.0,
                 bidirection=False, CheckGPS=None, metaData=None):
        localNodeDistance = {}
        mables = []
        edge_covered = {}

        if not newstyle:
            Queue = [(nodeid, -1, 0.0)]
        else:
            Queue = [(nid1, -1, float(dist1)), (nid2, -1, float(dist2))]

            # add samples along the starting tunnel between nid1 and nid2
            lat1 = self.nodes[nid1][0]
            lon1 = self.nodes[nid1][1]
            lat2 = self.nodes[nid2][0]
            lon2 = self.nodes[nid2][1]
            l = distance((lat2, lon2), (lat1, lon1))
            if l > 0:
                # sample alphas from 0..1
                step_alpha = step / l
                if step_alpha > 0:
                    alphas = np.arange(0.0, 1.0 + 1e-9, step_alpha)
                    for a in alphas:
                        latI = lat1 * a + lat2 * (1.0 - a)
                        lonI = lon1 * a + lon2 * (1.0 - a)
                        d1 = distance((latI, lonI), (lat1, lon1))
                        d2 = distance((latI, lonI), (lat2, lon2))
                        if dist1 - d1 < r or dist2 - d2 < r:
                            mables.append((latI, lonI, lat2 - lat1, lon2 - lon1))
                            if bidirection:
                                if (nid1 in self.nodeLink.get(nid2, [])) and (nid2 in self.nodeLink.get(nid1, [])):
                                    mables.append((latI + 0.00001, lonI + 0.00001, lat2 - lat1, lon2 - lon1))

        while Queue:
            node_cur, node_prev, dist_acc = Queue.pop(0)
            old_node_dist = localNodeDistance.get(node_cur, None)
            if old_node_dist is not None and old_node_dist <= dist_acc:
                continue
            if dist_acc > r:
                continue

            lat1 = self.nodes[node_cur][0]
            lon1 = self.nodes[node_cur][1]
            localNodeDistance[node_cur] = dist_acc

            reverseList = [] if direction else self.nodeLinkReverse.get(node_cur, [])
            visited_next_node = set()
            for next_node in self.nodeLink.get(node_cur, []) + list(reverseList):
                if next_node == node_prev or next_node == node_cur:
                    continue
                if next_node in visited_next_node:
                    continue
                visited_next_node.add(next_node)

                lat2 = self.nodes[next_node][0]
                lon2 = self.nodes[next_node][1]

                # segment length
                l = distance((lat2, lon2), (lat1, lon1))
                if l <= 0:
                    continue

                start_limitation = 0.0
                end_limitation = l
                key_forward = (node_cur, next_node)
                key_reverse = (next_node, node_cur)
                if key_forward in edge_covered:
                    start_limitation = edge_covered[key_forward]
                if key_reverse in edge_covered:
                    end_limitation = l - edge_covered[key_reverse]

                # bias so samples lie on the same global step grid relative to dist_acc
                bias = step * math.ceil(dist_acc / step) - dist_acc
                cur_start = max(bias, start_limitation)
                cur_end = min(end_limitation, r - dist_acc)

                # decide if this edge is a tunnel edge to be skipped (when metaData provided)
                turnnel_edge = False
                if metaData is not None:
                    nnn1 = self.nodeHashReverse[next_node]
                    nnn2 = self.nodeHashReverse[node_cur]
                    try:
                        if metaData.edgeProperty[metaData.edge2edgeid[(nnn1, nnn2)]]['layer'] < 0:
                            turnnel_edge = True
                    except Exception:
                        # if metadata missing, do not skip
                        turnnel_edge = False

                lat_arr, lon_arr = self._vectorized_samples_on_segment(lat1, lon1, lat2, lon2, cur_start, cur_end, step)
                if lat_arr is not None:
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    if not turnnel_edge:
                        for la, lo in zip(lat_arr.tolist(), lon_arr.tolist()):
                            mables.append((la, lo, dlat, dlon))
                            if bidirection:
                                if (next_node in self.nodeLink.get(node_cur, [])) and (node_cur in self.nodeLink.get(next_node, [])):
                                    mables.append((la + 0.00001, lo + 0.00001, dlat, dlon))

                # update coverage and continue BFS
                covered_val = cur_end if (lat_arr is None) else (cur_start + step * len(lat_arr))
                edge_covered[key_forward] = max(0.0, min(l, covered_val))
                Queue.append((next_node, node_cur, dist_acc + l))

        if CheckGPS is None:
            return mables
        result_marbles = []
        for m in mables:
            if CheckGPS(m[0], m[1]):
                result_marbles.append(m)
        return result_marbles

    def distanceBetweenTwoLocation(self, loc1, loc2, max_distance):
        # loc: (s, e, d_from_s, d_from_e)
        if loc1[0] == loc2[0] and loc1[1] == loc2[1]:
            return abs(loc1[2] - loc2[2])
        if loc1[0] == loc2[1] and loc1[1] == loc2[0]:
            return abs(loc1[2] - loc2[3])

        ans_dist = 1e9
        Queue = [(loc1[0], -1, loc1[2]), (loc1[1], -1, loc1[2])]
        localNodeDistance = {}

        while Queue:
            node_cur, node_prev, dist_acc = Queue.pop(0)
            old = localNodeDistance.get(node_cur, None)
            if old is not None and old <= dist_acc:
                continue
            if dist_acc > max_distance:
                continue
            localNodeDistance[node_cur] = dist_acc

            reverseList = self.nodeLinkReverse.get(node_cur, [])
            visited_next_node = set()
            for next_node in self.nodeLink.get(node_cur, []) + list(reverseList):
                if next_node == node_prev or next_node == node_cur:
                    continue
                if next_node in visited_next_node:
                    continue
                visited_next_node.add(next_node)

                if node_cur == loc2[0] and next_node == loc2[1]:
                    ans_dist = min(ans_dist, dist_acc + loc2[2])
                elif node_cur == loc2[1] and next_node == loc2[0]:
                    ans_dist = min(ans_dist, dist_acc + loc2[3])

                lat1 = self.nodes[node_cur][0]
                lon1 = self.nodes[node_cur][1]
                lat2 = self.nodes[next_node][0]
                lon2 = self.nodes[next_node][1]
                l = distance((lat2, lon2), (lat1, lon1))
                Queue.append((next_node, node_cur, dist_acc + l))

        return ans_dist


