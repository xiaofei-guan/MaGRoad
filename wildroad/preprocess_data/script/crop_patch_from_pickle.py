import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import LineString, box
import argparse
from skimage.draw import line
import os
from collections import defaultdict
from tqdm import tqdm

class GraphPatchCropper:
    """用于剪裁路网图和RGB图像的工具类，包含完善的边界处理逻辑"""
    
    def __init__(self, rgb_path, graph_path, edge_width=8, inner_offset=5, min_edge_length_ratio=0.1):
        """初始化剪裁器
        
        Args:
            rgb_path: RGB图像路径
            graph_path: 路网图pickle路径
            edge_width: 道路宽度
            inner_offset: 新生成边界点向内的偏移距离
            min_edge_length_ratio: 最小边长度比例，小于此比例的边将被过滤
        """
        self.rgb_path = rgb_path
        self.graph_path = graph_path
        self.edge_width = edge_width
        self.inner_offset = inner_offset
        self.min_edge_length_ratio = min_edge_length_ratio
        
        # 加载数据
        self.rgb_image = Image.open(rgb_path)
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
            
        # 用于存储处理结果
        self.cropped_image = None
        self.cropped_graph = None
        self.road_mask = None
        self.boundary_points = []  # 存储新生成的边界点
        
    def is_in_patch(self, point, patch):
        """检查点是否在patch内部
        
        Note: pickle数据坐标是(row, col)格式，patch是[left, top, right, bottom]
              row对应y方向(垂直)，col对应x方向(水平)
        """
        row, col = point
        # left <= col <= right (水平方向) and top <= row <= bottom (垂直方向)
        return (patch[0] <= col <= patch[2]) and (patch[1] <= row <= patch[3])
    
    def calculate_intersection(self, point1, point2, patch):
        """计算边与patch边界的交点
        
        Args:
            point1: 边的第一个点坐标 (row, col) - pickle数据格式
            point2: 边的第二个点坐标 (row, col)
            patch: 边界矩形 [left, top, right, bottom]
            
        Returns:
            交点坐标和对应的边界位置列表 [(row, col, boundary_type), ...]
            boundary_type: 'left', 'right', 'top', 'bottom'
        """
        # pickle数据是(row, col)格式，转换为shapely的(x, y)=(col, row)格式
        line = LineString([(point1[1], point1[0]), (point2[1], point2[0])])
        patch_box = box(patch[0], patch[1], patch[2], patch[3])
        
        # 如果线不与边界相交
        if not line.intersects(patch_box.boundary):
            return []
        
        # 计算与四条边界的交点
        left_bound = LineString([(patch[0], patch[1]), (patch[0], patch[3])])
        right_bound = LineString([(patch[2], patch[1]), (patch[2], patch[3])])
        top_bound = LineString([(patch[0], patch[1]), (patch[2], patch[1])])
        bottom_bound = LineString([(patch[0], patch[3]), (patch[2], patch[3])])
        
        boundaries = [
            (left_bound, 'left'),
            (right_bound, 'right'),
            (top_bound, 'top'),
            (bottom_bound, 'bottom')
        ]
        
        intersection_points = []
        for boundary, boundary_type in boundaries:
            if line.intersects(boundary):
                intersection = line.intersection(boundary)
                if not intersection.is_empty:
                    if intersection.geom_type == 'Point':
                        # shapely返回(x, y)=(col, row)格式，转换回(row, col)
                        intersection_points.append((intersection.y, intersection.x, boundary_type))
                    elif intersection.geom_type == 'MultiPoint':
                        for point in intersection.geoms:
                            intersection_points.append((point.y, point.x, boundary_type))
        
        return intersection_points
    
    def create_offset_point(self, intersection):
        """根据交点和边界类型，创建向内偏移的新点
        
        Args:
            intersection: 交点信息 (row, col, boundary_type)
            
        Returns:
            偏移后的新点坐标 (row, col)
        """
        row, col, b_type = intersection
        
        if b_type == 'left':
            # left边界，col向右偏移(增加)
            return (float(row), float(col + self.inner_offset))
        elif b_type == 'right':
            # right边界，col向左偏移(减少)
            return (float(row), float(col - self.inner_offset))
        elif b_type == 'top':
            # top边界，row向下偏移(增加)
            return (float(row + self.inner_offset), float(col))
        elif b_type == 'bottom':
            # bottom边界，row向上偏移(减少)
            return (float(row - self.inner_offset), float(col))
        
        return (float(row), float(col))  # 默认不偏移
    
    def calculate_distance(self, point1, point2):
        """计算两点之间的欧几里得距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def process_edge_one_inside(self, node, neighbor, patch, offset_x, offset_y, node_rel, new_edges):
        """处理一个点在patch内，一个点在patch外的边
        
        Args:
            node: patch内的点坐标
            neighbor: patch外的点坐标
            patch: 剪裁区域 [left, top, right, bottom]
            offset_x, offset_y: 坐标偏移量
            node_rel: 内部点的相对坐标 (float, float)
            new_edges: 新边的集合

        Returns:
            生成的边界点在patch内的相对坐标
        """
        # 计算边与边界的交点
        intersections = self.calculate_intersection(node, neighbor, patch)
        if not intersections:
            return None
            
        # 对于一个点在内部一个点在外部的情况，应该只有一个交点
        intersection = intersections[0]
        
        # 计算内部点与交点之间的距离
        distance_to_intersection = self.calculate_distance(node, (intersection[0], intersection[1]))
        
        # 如果距离太小，不处理这条边
        patch_width = patch[2] - patch[0]
        if distance_to_intersection < self.min_edge_length_ratio * patch_width:
            return None
        
        # 创建新的边界点（向内偏移）
        new_point = self.create_offset_point(intersection)
        # new_point是(row, col)，offset_x=left, offset_y=top
        # row_rel = row - top, col_rel = col - left
        new_point_rel = (float(new_point[0] - offset_y), float(new_point[1] - offset_x))
        
        # 记录边界点信息（用于可视化）
        self.boundary_points.append((new_point_rel, node_rel))
        
        # 添加到节点的邻居列表
        self.cropped_graph[node_rel].append(new_point_rel)
        
        # 记录新边界点及其邻居
        new_edges.append((new_point_rel, [node_rel]))
        
        return new_point_rel
    
    def process_edge_both_outside(self, point1, point2, patch, offset_x, offset_y, new_edges):
        """处理两个点都在patch外但边穿过patch的情况
        
        Args:
            point1, point2: 边的两个端点坐标
            patch: 剪裁区域 [left, top, right, bottom]
            offset_x, offset_y: 坐标偏移量
            new_edges: 新边的集合

        Returns:
            如果成功处理，返回True；否则返回False
        """
        # 计算边与边界的所有交点
        intersections = self.calculate_intersection(point1, point2, patch)
        
        # 如果没有交点或只有一个交点（切线），忽略这条边
        if len(intersections) < 2:
            return
        
        # 获取两个交点
        int1, int2 = intersections[:2]
        
        # 计算两个交点之间的距离
        distance_between_intersections = self.calculate_distance(
            (int1[0], int1[1]), 
            (int2[0], int2[1])
        )
        
        # 如果两个交点距离太近，忽略这条边
        patch_width = patch[2] - patch[0]
        if distance_between_intersections < self.min_edge_length_ratio * patch_width:
            return
        
        # 创建两个新的边界点（向内偏移）
        new_point1 = self.create_offset_point(int1)
        new_point2 = self.create_offset_point(int2)
        
        # 转换为相对坐标，new_point是(row, col)，offset_x=left, offset_y=top
        # row_rel = row - top, col_rel = col - left
        new_point1_rel = (float(new_point1[0] - offset_y), float(new_point1[1] - offset_x))
        new_point2_rel = (float(new_point2[0] - offset_y), float(new_point2[1] - offset_x))
        
        # 记录边界点信息（用于可视化）
        self.boundary_points.append((new_point1_rel, new_point2_rel))
        self.boundary_points.append((new_point2_rel, new_point1_rel))
        
        # 为两个新点之间添加边
        new_edges.append((new_point1_rel, [new_point2_rel]))
        new_edges.append((new_point2_rel, [new_point1_rel]))
        
    def crop_patch(self, patch):
        """剪裁指定区域的图像和路网
        
        Args:
            patch: 剪裁区域 [left, top, right, bottom]
            
        Returns:
            cropped_image: 剪裁后的RGB图像
            cropped_graph: 剪裁后的路网图
            road_mask: 道路掩码
        """
        # 剪裁RGB图像
        self.cropped_image = self.rgb_image.crop(patch)
        
        # 初始化mask
        mask_size = (patch[2] - patch[0], patch[3] - patch[1])
        self.road_mask = np.zeros(mask_size[::-1], dtype=np.uint8)  # (height, width)
        
        # 初始化剪裁后的路网图
        self.cropped_graph = {}
        self.boundary_points = []
        
        # 临时存储需要添加的新边
        new_edges = []
        
        # 为转换坐标准备偏移量
        offset_x, offset_y = patch[0], patch[1]
        
        # 第一步：处理节点在patch内部的情况
        for node, neighbors in self.graph.items():
            # 如果节点在patch内部
            if self.is_in_patch(node, patch):
                # node是(row, col)格式，offset_x=left, offset_y=top
                # row_rel = row - top, col_rel = col - left
                node_rel = (float(node[0] - offset_y), float(node[1] - offset_x))
                
                # 初始化该节点在新图中的邻居列表
                self.cropped_graph[node_rel] = []
                
                # 处理每个邻居节点
                for neighbor in neighbors:
                    # 如果邻居也在patch内
                    if self.is_in_patch(neighbor, patch):
                        neighbor_rel = (float(neighbor[0] - offset_y), float(neighbor[1] - offset_x))
                        self.cropped_graph[node_rel].append(neighbor_rel)
                    else:
                        # 一个点在内部，一个点在外部
                        self.process_edge_one_inside(node, neighbor, patch, offset_x, offset_y, node_rel, new_edges)
        
        # 第二步：处理两个点都在patch外部但边穿过patch的情况
        processed_edges = set()  # 避免重复处理同一条边
        
        for node, neighbors in self.graph.items():
            # 只处理patch外的点
            if self.is_in_patch(node, patch):
                continue
                
            for neighbor in neighbors:
                # 如果邻居也在patch外
                if not self.is_in_patch(neighbor, patch):
                    # 创建规范化的边ID（始终将较小的节点放在前面）
                    edge_id = tuple(sorted([node, neighbor]))
                    
                    # 如果已经处理过这条边，跳过
                    if edge_id in processed_edges:
                        continue
                        
                    processed_edges.add(edge_id)
                    
                    # 处理两个点都在外部的边
                    self.process_edge_both_outside(node, neighbor, patch, offset_x, offset_y, new_edges)
        
        # 添加新的边界点到图中
        for new_node, neighbors in new_edges:
            if new_node not in self.cropped_graph:
                self.cropped_graph[new_node] = []
            
            for neighbor in neighbors:
                if neighbor not in self.cropped_graph[new_node]:
                    self.cropped_graph[new_node].append(neighbor)
                
        # 创建道路mask
        self.create_road_mask()
        
        # 过滤度为0的节点（没有邻居的孤立点）
        self.filter_isolated_nodes()
        
        return self.cropped_image, self.cropped_graph, self.road_mask
    
    def filter_isolated_nodes(self):
        """过滤图中度为0的孤立节点"""
        # 找出所有度为0的节点
        isolated_nodes = [node for node, neighbors in self.cropped_graph.items() if len(neighbors) == 0]
        
        # 删除这些节点
        for node in isolated_nodes:
            del self.cropped_graph[node]
            
        if isolated_nodes:
            print(f"已移除 {len(isolated_nodes)} 个孤立节点")
    
    def create_road_mask(self):
        """基于剪裁后的图创建道路mask
        
        Note: cropped_graph中的节点是(row_rel, col_rel)格式
              numpy数组索引是[row, col]
              skimage.draw.line期望(r0, c0, r1, c1)
        """
        for node, neighbors in self.cropped_graph.items():
            for neighbor in neighbors:
                
                # node和neighbor是(row_rel, col_rel)格式
                r0, c0 = map(int, node)
                r1, c1 = map(int, neighbor)
                # skimage.draw.line期望(row, col)格式
                rr, cc = line(r0, c0, r1, c1)
                
                # 扩展道路宽度
                if self.edge_width > 1:
                    r = np.clip(rr, 0, self.road_mask.shape[0]-1)
                    c = np.clip(cc, 0, self.road_mask.shape[1]-1)
                    for dr in range(-int(self.edge_width//2), int(self.edge_width//2)+1):
                        for dc in range(-int(self.edge_width//2), int(self.edge_width//2)+1):
                            nr = np.clip(r + dr, 0, self.road_mask.shape[0]-1)
                            nc = np.clip(c + dc, 0, self.road_mask.shape[1]-1)
                            self.road_mask[nr, nc] = 1
                else:
                    self.road_mask[rr, cc] = 1
    
    def visualize(self, show=True, save_path=None):
        """可视化剪裁结果
        
        Args:
            show: 是否显示图像
            save_path: 保存路径，如果为None则不保存
        """
        if self.cropped_image is None or self.cropped_graph is None:
            raise ValueError("先调用crop_patch方法进行剪裁")
        
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 绘制RGB上的路网
        ax1.imshow(self.cropped_image)
        ax1.set_title('Road Network on RGB')
        
        # 绘制路网边 - 节点是(row, col)格式，matplotlib使用(x, y)=(col, row)格式
        for node, neighbors in self.cropped_graph.items():
            for neighbor in neighbors:
                # node = (row, col), matplotlib plot需要(x, y) = (col, row)
                ax1.plot([node[1], neighbor[1]], [node[0], neighbor[0]], 'g-', linewidth=3)
        
        # 绘制普通节点
        if self.cropped_graph:
            node_coords = list(self.cropped_graph.keys())
            # 节点是(row, col)格式，转换为matplotlib的(x, y) = (col, row)
            plot_coords = [(col, row) for row, col in node_coords]
            ax1.scatter(*zip(*plot_coords), c='red', s=30, zorder=3)
        
        # 高亮边界节点
        if self.boundary_points:
            boundary_coords = [bp[0] for bp in self.boundary_points]
            # 节点是(row, col)格式，转换为matplotlib的(x, y) = (col, row)
            boundary_plot_coords = [(col, row) for row, col in boundary_coords]
            ax1.scatter(*zip(*boundary_plot_coords), c='blue', s=45, marker='s', zorder=4, label='point on boundary')
        
        ax1.legend()
        
        # 绘制道路mask
        ax2.imshow(self.road_mask, cmap='gray')
        ax2.set_title('Road Mask')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
            
    def save_results(self, output_dir, patch, save_boundary_points=False):
        """保存剪裁结果
        
        Args:
            output_dir: 输出目录
            patch: 剪裁区域
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        name_prefix = f"cropped_{patch[0]}_{patch[1]}_{patch[2]}_{patch[3]}"
        # 保存RGB图像
        rgb_output = os.path.join(output_dir, name_prefix + '_rgb.png')
        self.cropped_image.save(rgb_output)
        
        # 保存道路mask
        mask_output = os.path.join(output_dir, name_prefix + '_mask.png')
        Image.fromarray(self.road_mask * 255).save(mask_output)
        
        # 保存剪裁后的图为pickle格式
        graph_output = os.path.join(output_dir, name_prefix + '_graph.pickle')
        with open(graph_output, 'wb') as f:
            pickle.dump(self.cropped_graph, f)
            
        # 保存边界点信息为pickle格式
        if save_boundary_points:
            boundary_output = os.path.join(output_dir, name_prefix + '_boundary_points.pickle')
            boundary_dict = {bp[0]: bp[1] for bp in self.boundary_points}
            with open(boundary_output, 'wb') as f:
                pickle.dump(boundary_dict, f)
            
        # 保存可视化图像
        # vis_output = os.path.join(output_dir, name_prefix + '_visualization.png')
        # self.visualize(show=False, save_path=vis_output)

def compute_connectivity_stats(graph):
    """计算路网的连通性统计信息"""
    degree_dist = defaultdict(int)
    total_nodes = len(graph)
    total_edges = 0
    
    for node, neighbors in graph.items():
        degree = len(neighbors)
        degree_dist[degree] += 1
        total_edges += degree
    
    # 每条边被计算了两次
    total_edges //= 2
    
    stats = {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "avg_degree": total_edges * 2 / total_nodes if total_nodes > 0 else 0,
        "degree_distribution": dict(degree_dist)
    }
    
    return stats

def generate_patches(left, top, right, bottom, patch_size, num_patches_rows=None, num_patches_cols=None):
    """在指定区域内生成多个正方形patch的坐标，支持patch之间重叠以覆盖整个区域
    
    Args:
        left: 区域左边界
        top: 区域上边界  
        right: 区域右边界
        bottom: 区域下边界
        patch_size: 每个patch的大小（正方形）
        num_patches_rows: 行方向上的patch数量
        num_patches_cols: 列方向上的patch数量
        
    Returns:
        patch列表，每个patch为 [left, top, right, bottom]
    """
    patches = []
    
    # 计算区域的宽度和高度
    region_width = right - left
    region_height = bottom - top
    
    # 确保num_patches_rows和num_patches_cols有值
    if num_patches_rows is None:
        num_patches_rows = (region_height + patch_size - 1) // patch_size  # 向上取整
    if num_patches_cols is None:
        num_patches_cols = (region_width + patch_size - 1) // patch_size  # 向上取整
    
    # 计算行和列方向上的步长，使得patch均匀分布并覆盖整个区域
    if num_patches_rows == 1:
        row_stride = 0  # 只有一行不需要步长
    else:
        row_stride = (region_height - patch_size) / (num_patches_rows - 1)
    
    if num_patches_cols == 1:
        col_stride = 0  # 只有一列不需要步长
    else:
        col_stride = (region_width - patch_size) / (num_patches_cols - 1)
    
    # 生成所有patch的坐标
    for i in range(num_patches_rows):
        for j in range(num_patches_cols):
            # 计算patch的左上角坐标
            patch_left = min(int(left + j * col_stride), right - patch_size)
            patch_top = min(int(top + i * row_stride), bottom - patch_size)
            
            # 确保最后一行/列的patch不会超出区域边界
            patch_right = min(patch_left + patch_size, right)
            patch_bottom = min(patch_top + patch_size, bottom)
            
            patches.append([patch_left, patch_top, patch_right, patch_bottom])
    
    return patches

def main():
    parser = argparse.ArgumentParser(description="剪裁路网图和RGB图像")
    parser.add_argument("rgb_path", help="RGB图像路径")
    parser.add_argument("graph_path", help="路网图pickle路径")
    parser.add_argument("--output", "-o", default="./output", help="输出目录")
    parser.add_argument("--patch", nargs=4, type=int, help="剪裁区域 [left top right bottom]")
    parser.add_argument("--region", nargs=4, type=int, help="生成patch的区域范围 [left top right bottom]")
    parser.add_argument("--patch_size", type=int, help="每个patch的大小（正方形）")
    parser.add_argument("--num_patches_rows", type=int, help="行方向上的patch数量")
    parser.add_argument("--num_patches_cols", type=int, help="列方向上的patch数量")
    parser.add_argument("--edge_width", type=int, default=4, help="道路宽度")
    parser.add_argument("--inner_offset", type=int, default=5, help="边界点向内偏移距离")
    parser.add_argument("--min_edge_ratio", type=float, default=0.08, help="最小边长度比例")
    
    args = parser.parse_args()
    
    # 判断操作模式：单个patch还是多个patch
    patches = []
    
    if args.patch:
        # 单个patch模式
        patches = [args.patch]
    elif args.patch_size and args.region:
        # 多个patch模式，在指定区域内生成
        patches = generate_patches(
            args.region[0], args.region[1], args.region[2], args.region[3],
            args.patch_size, 
            args.num_patches_rows, 
            args.num_patches_cols
        )
        
        # 打印生成的patch信息
        region_width = args.region[2] - args.region[0]
        region_height = args.region[3] - args.region[1]
        num_rows = args.num_patches_rows or ((region_height + args.patch_size - 1) // args.patch_size)
        num_cols = args.num_patches_cols or ((region_width + args.patch_size - 1) // args.patch_size)
        print(f"在区域 {args.region} 内生成了 {len(patches)} 个patch (行: {num_rows}, 列: {num_cols})，每个大小为 {args.patch_size}x{args.patch_size}")
    elif args.patch_size:
        # 如果没有指定区域，使用整张图像
        rgb_image = Image.open(args.rgb_path)
        image_width, image_height = rgb_image.size
        patches = generate_patches(
            0, 0, image_width, image_height,
            args.patch_size, 
            args.num_patches_rows, 
            args.num_patches_cols
        )
        print(f"在整张图像内生成了 {len(patches)} 个patch，每个大小为 {args.patch_size}x{args.patch_size}")
    else:
        # 默认使用整张图像
        rgb_image = Image.open(args.rgb_path)
        image_width, image_height = rgb_image.size
        patches = [[0, 0, image_width, image_height]]
    
    # 创建剪裁器
    cropper = GraphPatchCropper(
        args.rgb_path, 
        args.graph_path,
        edge_width=args.edge_width,
        inner_offset=args.inner_offset,
        min_edge_length_ratio=args.min_edge_ratio
    )
    
    # 输出原始路网统计信息
    print("原始路网统计:")
    original_stats = compute_connectivity_stats(cropper.graph)
    print(original_stats)
    
    # 使用tqdm创建进度条
    for patch in tqdm(patches, desc="处理patch", unit="patch"):
        patch_name = f"{patch[0]}_{patch[1]}_{patch[2]}_{patch[3]}"
        try:
            # 执行剪裁
            cropped_image, cropped_graph, road_mask = cropper.crop_patch(patch)
            
            # 如果是单个patch模式，显示剪裁后路网统计
            if len(patches) == 1:
                print("\n剪裁后路网统计:")
                cropped_stats = compute_connectivity_stats(cropped_graph)
                print(cropped_stats)
                
                # 可视化
                cropper.visualize(show=True, save_path=os.path.join(args.output, f"{patch_name}_visualization.png"))
            
            # 保存结果
            cropper.save_results(args.output, patch)
            
        except Exception as e:
            print(f"处理patch {patch_name} 时出错: {str(e)}")
            continue

        print(f"结果已保存到 {args.output}")

if __name__ == "__main__":
    main()
    # 单个patch调用示例:
    # python crop_patch_from_pickle.py map.jpg road_network.pkl --patch 5888 8445 6912 9469 --edge_width 4 --inner_offset 5 --min_edge_ratio 0.05

    # 在指定区域内生成多个patch调用示例:
    # python crop_patch_from_pickle.py map.jpg road_network.pkl --output ./map_patches --region 0 0 8192 4448 --patch_size 1024 --num_patches_rows 4 --num_patches_cols 8 --edge_width 4 --inner_offset 5 --min_edge_ratio 0.08

    # 在整张图像内生成多个patch调用示例:
    # python crop_patch_from_pickle.py map.jpg road_network.pkl --output ./map_patches --patch_size 1024 --num_patches_rows 4 --num_patches_cols 8 --edge_width 4 --inner_offset 5 --min_edge_ratio 0.08

    # xy format