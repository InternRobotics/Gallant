import numpy as np
import trimesh
from shapely.geometry import box as shapely_box
import random
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.terrains import (
    SubTerrainBaseCfg,
    MeshPyramidStairsTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg
)
from isaaclab.utils import configclass
from isaaclab.terrains.trimesh.utils import make_border

from active_adaptation.envs.terrain import BetterTerrainImporter
from active_adaptation.registry import Registry

from .terrain_generator import hussar_terrain_generator
from .terrain_importer import hussar_terrain_importer
from .functional import hussar_pillars_terrain

registry = Registry.instance()


TREE_INFOS = []


def hussar_tree_terrain(
    difficulty: float, cfg: "HussarTreeCfg", num_row: int, num_col: int
)-> tuple[list[trimesh.Trimesh], np.ndarray]:
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.3)
    size_x, size_y = cfg.size
    x0 = [size_x, size_y, 0]
    x1 = [size_x, 0.0, 0]
    x2 = [0.0, size_y, 0]
    x3 = [0.0, 0.0, 0]
    min_gap = cfg.min_gap
    max_gap = cfg.max_gap
    # generate the tri-mesh with two triangles
    vertices = np.array([x0, x1, x2, x3])
    faces = np.array([[1, 0, 2], [2, 3, 1]])
    plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    combined_meshes = [plane_mesh]
    current_gap = max_gap - difficulty * (max_gap - min_gap)    
    # 计算树干的数量和分布
    # print("difficulty:", difficulty)
    tree_density = 0.2 + difficulty * 0.3  # 随难度增加树的密度
    area = size_x * size_y
    num_trees = int(area * tree_density / (current_gap * current_gap))
    
    # 限制最大树数量，避免过度密集
    num_trees = min(num_trees, 200)
    
    # 跟踪已放置的树位置
    tree_positions = []
    
    for _ in range(num_trees):
        # 生成候选位置
        attempt = 0
        valid_position = False
        
        while attempt < 50 and not valid_position:  # 最多尝试50次找位置
            x = random.uniform(0.0 + 0.5, size_x - 0.5)
            y = random.uniform(0.0 + 0.5, size_y - 0.5)
            if (x - size_x/2)**2 + (y - size_y/2)**2 < 2:
                continue
            x_near_boudary = (x < 0.8 or x > size_x - 0.8)
            y_near_boudary = (y < 0.8 or y > size_y - 0.8)
            if x_near_boudary or y_near_boudary:
                continue
            # 检查与其他树的距离
            valid_position = True
            for pos in tree_positions:
                dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                if dist < current_gap:
                    valid_position = False
                    break
                    
            attempt += 1
        
        if valid_position:
            tree_positions.append((x, y))
            
            # 树干参数
            trunk_height = random.uniform(0.8, 2.0)
            trunk_radius = random.uniform(0.1, 0.3)
            
            # 创建树干
            trunk = trimesh.creation.cylinder(
                radius=trunk_radius,
                height=trunk_height,
                sections=8
            )
            
            # 随机倾斜角度 (0-20度)
            tilt_angle = random.uniform(0, 15) * (np.pi / 180) * 0.0 # NOTE directly use 0 here
            tilt_direction = random.uniform(0, 2 * np.pi)
            
            # 计算倾斜轴和矩阵
            tilt_axis = [np.cos(tilt_direction), np.sin(tilt_direction), 0]
            tilt_matrix = trimesh.transformations.rotation_matrix(tilt_angle, tilt_axis)
            
            # 应用变换
            bottom_to_origin = trimesh.transformations.translation_matrix([0, 0, trunk_height/2])
            translation = trimesh.transformations.translation_matrix([x, y, 0])
            
            trunk.apply_transform(bottom_to_origin)
            trunk.apply_transform(tilt_matrix)
            trunk.apply_transform(translation)
            combined_meshes.append(trunk)
            x = x + 0. * np.sin(tilt_angle) * np.cos(tilt_direction) + ((num_row  - 5.0) * cfg.size[0])
            y = y + 0. * np.sin(tilt_angle) * np.sin(tilt_direction) + ((num_col - 10.0) * cfg.size[1])
            z = 0.75 * np.cos(tilt_angle)
            
            TREE_INFOS.append([x, y, z])
            
    if len(combined_meshes) > 1:
        combined_mesh = trimesh.util.concatenate(combined_meshes)
    else:
        combined_mesh = combined_meshes[0]
    return [combined_mesh], origin

def hussar_door_terrain(
    difficulty: float, cfg: "HussarDoorCfg", num_row: int, num_col: int
):
    """
    同一 difficulty 内部完全一致（无随机），按“米制长度”控制的同心环 + 实体门：
    - 每圈均匀重复：墙弧(分隔, 线长=sep_len_m) + 门口(线长=door_width_m)。
    - 门口附带两侧“门框/立柱”（径向矩形板），非空洞缝隙。
    - 环距 ring_spacing 固定（同一 difficulty），环与环完全一致的参数，只是半径不同导致每圈门数量按周长自适应。
    - TREE_INFOS：墙弧沿切向采样；门框沿径向采样；保留平铺偏移。
    """
    global TREE_INFOS

    # ===== 地面平面 =====
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.3)
    size_x, size_y = cfg.size
    cx, cy = size_x * 0.5, size_y * 0.5

    plane = trimesh.Trimesh(
        vertices=np.array([[size_x, size_y, 0], [size_x, 0, 0], [0, size_y, 0], [0, 0, 0]]),
        faces=np.array([[1, 0, 2], [2, 3, 1]])
    )
    combined_meshes = [plane]

    # ===== 统一参数（仅由 difficulty 决定；同一 difficulty 内对所有环一致）=====
    margin = 0.8
    wall_thickness = 0.10                         # 墙厚（径向方向）
    wall_height = 1.6 + 0.6 * difficulty          # 墙高

    # 圈距（所有环相同）
    ring_spacing = float(np.clip(1.5 - 0.35 * difficulty, 0.6, 1.2))
    min_radius   = 1.1 + 0.5 * difficulty

    # 门宽（米制），随难度从 1.6 → 0.8 线性收窄（difficulty∈[0,1]）
    door_width_m = float(np.clip(1.6 - difficulty * (1.6 - 0.8), 0.8, 1.6))

    # 门间隔（墙弧线长，米制）：沿用你原本的 min/max gap 插值
    current_gap = cfg.max_gap - difficulty * (cfg.max_gap - cfg.min_gap)  # == sep_len_m
    sep_len_m   = float(np.clip(current_gap, 0.4, 4.0))

    # 近似圆弧的单板目标弧长（影响圆滑程度，所有环一致）
    target_panel_arc = 0.40

    # 门框（立柱）几何：沿“切向”为厚度，沿“径向”为深度
    jamb_tangential_thickness = 0.10              # 门框在切向方向的可见厚度（越大越显眼）
    jamb_height = wall_height                     # 与墙等高
    # 可选：上梁如果想做“真正的门框”，可以再加一块高位横梁，但容易挡路，这里不加。

    # 采样（TREE_INFOS）设置
    tangent_sample_spacing = 0.35                 # 墙弧：沿切向采样步长
    radial_sample_spacing  = 0.25                 # 门框：沿径向采样步长
    sample_height          = 0.75                 # 所有采样点 z

    # 外半径限制
    max_radius_allowed = min(size_x, size_y) * 0.5 - margin - wall_thickness

    # ===== 工具：添加墙弧小板（切向）=====
    def add_wall_panel(r, theta, chord_len):
        # 局部坐标：X=径向厚度，Y=切向长度，Z=高度
        box = trimesh.creation.box(extents=[wall_thickness, chord_len, wall_height])
        # 落地
        T0 = np.eye(4); T0[:3, 3] = [0.0, 0.0, wall_height / 2.0]
        box.apply_transform(T0)
        # 旋转：局部X→径向(cosθ,sinθ,0)，局部Y→切向(-sinθ,cosθ,0)
        Rz = trimesh.transformations.rotation_matrix(theta, [0, 0, 1.0])
        box.apply_transform(Rz)
        # 平移到圆周
        px = cx + r * np.cos(theta)
        py = cy + r * np.sin(theta)
        box.apply_transform(trimesh.transformations.translation_matrix([px, py, 0.0]))
        combined_meshes.append(box)

        # TREE_INFOS：沿切向采样
        half_L = chord_len / 2.0
        n_half = max(1, int(half_L / tangent_sample_spacing))
        ts = [0.0]
        for k in range(1, n_half + 1):
            t = k * tangent_sample_spacing
            if t < half_L - 1e-9:
                ts.extend([+t, -t])
        ut = np.array([-np.sin(theta), np.cos(theta)])  # 切向单位向量
        for t in ts:
            sx = px + ut[0] * t
            sy = py + ut[1] * t
            x_w = sx + ((num_row - 5.0) * size_x)
            y_w = sy + ((num_col - 10.0) * size_y)
            TREE_INFOS.append([float(x_w), float(y_w), float(sample_height)])

    # ===== 工具：添加门框立柱（径向）=====
    def add_jamb(r, theta_edge):
        """
        在门口边界角度 theta_edge 放一根“径向立柱”：
        - 局部X=径向深度(=wall_thickness)，局部Y=切向厚度(=jamb_tangential_thickness)
        - 位置在中心半径 r（与墙中心一致）
        - 沿径向采样 TREE_INFOS（更符合门边界的真实影响）
        """
        box = trimesh.creation.box(extents=[wall_thickness, jamb_tangential_thickness, jamb_height])
        T0 = np.eye(4); T0[:3, 3] = [0.0, 0.0, jamb_height / 2.0]
        box.apply_transform(T0)
        Rz = trimesh.transformations.rotation_matrix(theta_edge, [0, 0, 1.0])
        box.apply_transform(Rz)
        px = cx + r * np.cos(theta_edge)
        py = cy + r * np.sin(theta_edge)
        box.apply_transform(trimesh.transformations.translation_matrix([px, py, 0.0]))
        combined_meshes.append(box)

        # TREE_INFOS：沿“径向”采样（抓住门边线）
        half_R = wall_thickness / 2.0
        n_half = max(1, int(half_R / radial_sample_spacing))
        ts = [0.0]
        for k in range(1, n_half + 1):
            t = k * radial_sample_spacing
            if t < half_R - 1e-9:
                ts.extend([+t, -t])
        ur = np.array([np.cos(theta_edge), np.sin(theta_edge)])  # 径向单位向量
        for t in ts:
            sx = px + ur[0] * t
            sy = py + ur[1] * t
            x_w = sx + ((num_row - 5.0) * size_x)
            y_w = sy + ((num_col - 10.0) * size_y)
            TREE_INFOS.append([float(x_w), float(y_w), float(sample_height)])

    # ===== 工具：在角区间 [th0, th1] 铺墙弧 =====
    def fill_wall_arc(r, th0, th1):
        if th1 < th0:
            th1 += 2 * np.pi
        L = th1 - th0
        # 角步长由目标弧长→角度：dθ ≈ target_panel_arc / r
        dtheta = float(np.clip(target_panel_arc / max(r, 1e-6), 0.05, 0.30))
        n_seg = max(1, int(np.ceil(L / dtheta)))
        dtheta = L / n_seg
        for k in range(n_seg):
            theta_mid = th0 + (k + 0.5) * dtheta
            chord_len = float(max(0.12, 2.0 * r * np.sin(0.5 * dtheta)))
            add_wall_panel(r, theta_mid, chord_len)

    # ===== 逐圈生成（同一 difficulty 内“统一米制参数”）=====
    r = min_radius
    if r + wall_thickness > max_radius_allowed:
        combined_mesh = trimesh.util.concatenate(combined_meshes)
        return [combined_mesh], origin

    while r + wall_thickness <= max_radius_allowed:
        # 把“米制长度”转换为该半径上的角度
        sep_ang  = sep_len_m  / max(r, 1e-6)
        door_ang = door_width_m / max(r, 1e-6)

        unit_ang = sep_ang + door_ang
        if unit_ang <= 1e-6:
            break

        # 能放下的整单位个数（每单位=墙弧+门）
        slots = int(np.floor((2.0 * np.pi) / unit_ang))
        slots = max(1, min(slots, 512))

        # 重新均匀化角步长，让 slots*unit_ang_exact 正好 = 2π
        unit_ang_exact = (2.0 * np.pi) / slots
        # 按相同比例分配到墙弧与门：保持米制比例
        ratio = sep_len_m / (sep_len_m + door_width_m)
        sep_ang_exact  = unit_ang_exact * ratio
        door_ang_exact = unit_ang_exact * (1.0 - ratio)

        theta = -np.pi  # 固定相位，确定性
        for _ in range(slots):
            # 1) 墙弧：[theta, theta + sep_ang_exact]
            th0 = theta
            th1 = theta + sep_ang_exact
            fill_wall_arc(r, th0, th1)

            # 2) 门框（两侧立柱）：在门口两侧角度 th1 与 th1+door_ang_exact
            left_edge  = th1
            right_edge = th1 + door_ang_exact
            add_jamb(r, left_edge)
            add_jamb(r, right_edge)

            # 3) 门口区间 [th1, th1+door_ang_exact] 留空（可选上梁，这里不放，避免挡路）

            theta = right_edge  # 进入下一个单位

        # 下一圈（环距恒定，同一 difficulty 下一直一样）
        r += ring_spacing

    combined_mesh = trimesh.util.concatenate(combined_meshes)
    return [combined_mesh], origin

import numpy as np
import trimesh
from shapely.geometry import box as shapely_box

def hussar_platform_terrain(
    difficulty: float, cfg: "HussarPlatformCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    Generates concentric raised ring platforms. Each ring is extruded upward to `box_height`.
    The rest of the plane stays at z = 0 (no excavation).

    - Ring thickness is fixed at 0.6 (same as original).
    - Spacing between rings (gap) scales with difficulty via `step_gap`.
    """
    # Heights and spacing derived from difficulty
    box_height = cfg.min_height + difficulty * (cfg.max_height - cfg.min_height)
    step_gap   = cfg.min_gap   + difficulty * (cfg.max_gap   - cfg.min_gap)
    ring_thickness = 0.6

    # How many rings fit (use your original margin logic: 1.0 inner margin + 1.2 outer)
    size_x, size_y = cfg.size
    num_plats_x = (size_x - 1.0 - 1.2) // (step_gap + ring_thickness) + 1
    num_plats_y = (size_y - 1.0 - 1.2) // (step_gap + ring_thickness) + 1
    num_steps = int(min(num_plats_x, num_plats_y))

    # Base plane at z = 0 (unchanged)
    # 覆盖整个 cfg.size 的地面平面（z = 0）
    vertices = np.array([
        [0.0,      0.0,     0.0],
        [size_x,   0.0,     0.0],
        [0.0,      size_y,  0.0],
        [size_x,   size_y,  0.0],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=int)
    plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    combined_meshes = [plane_mesh]

    # Build concentric raised ring platforms, all extruded upwards by box_height
    # Inner edge starts at radius-like "half-width" of 1.0, then grows by (step_gap + ring_thickness)
    for t in range(max(0, num_steps)):
        inner_w = 1.0 + t * (step_gap + ring_thickness)
        outer_w = inner_w + ring_thickness

        # Guard: stop if ring would exceed the terrain bounds
        if (outer_w * 2.0) >= min(size_x, size_y):
            break

        outer_box = shapely_box(size_x / 2 - outer_w,
                                size_y / 2 - outer_w,
                                size_x / 2 + outer_w,
                                size_y / 2 + outer_w)
        inner_box = shapely_box(size_x / 2 - inner_w,
                                size_y / 2 - inner_w,
                                size_x / 2 + inner_w,
                                size_y / 2 + inner_w)
        ring_poly = outer_box.difference(inner_box)

        # Extrude ring upward (z in [0, box_height])
        ring_mesh = trimesh.creation.extrude_polygon(ring_poly, height=box_height)
        # No translation needed; base sits at z = 0.
        combined_meshes.append(ring_mesh)

    # Merge all meshes
    combined_mesh = trimesh.util.concatenate(combined_meshes) if len(combined_meshes) > 1 else combined_meshes[0]

    # Origin: keep your original spawn height
    origin = (size_x / 2.0, size_y / 2.0, 0.3)
    return [combined_mesh], np.array(origin, dtype=float)

def hussar_ceil_terrain(
    difficulty: float, cfg: "HussarCeilCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    ceil_max_height = cfg.ceil_max_height
    ceil_min_height = cfg.ceil_min_height
    size = cfg.size
    ceiling_height = ceil_max_height - difficulty * (ceil_max_height - ceil_min_height)
    origin = (size[0] / 2.0, size[1] / 2.0, 0.3)
    size_x, size_y = size
    # 创建天花板障碍物
    x0 = [size_x, size_y, 0]
    x1 = [size_x, 0.0, 0]
    x2 = [0.0, size_y, 0]
    x3 = [0.0, 0.0, 0]
    # generate the tri-mesh with two triangles
    vertices = np.array([x0, x1, x2, x3])
    faces = np.array([[1, 0, 2], [2, 3, 1]])
    num_ceiling_parts = int(10 + difficulty * 30)  # 随难度增加天花板数量
    plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    combined_meshes = [plane_mesh]
    for _ in range(num_ceiling_parts):
        # 随机确定天花板大小
        width = random.uniform(1.0, 2.0)
        depth = random.uniform(1.0, 2.0)
        height = random.uniform(0.05, 0.15)
        
        flag = True
        while flag:
            x = random.uniform(0 + width/2, size_x + width/2)
            y = random.uniform(0 + depth/2, size_y - depth/2)
            flag = ((x > size_x/2 - 0.75 - width/2) & (x < size_x/2 + 0.75 + width/2)) & ((y > size_y/2 - 0.75 - depth/2) & (y < size_y/2 + 0.75 + depth / 2))
            

        # 高度在计算的范围内随机浮动一点
        z = ceiling_height + random.uniform(-0.1, 0.1) + height/2
        
        # 创建天花板块
        ceiling_part = trimesh.creation.box(
            extents=[width, depth, height]
        )
        
        # 应用平移
        translation = trimesh.transformations.translation_matrix([x, y, z])
        ceiling_part.apply_transform(translation)
        
        combined_meshes.append(ceiling_part)
    if len(combined_meshes) > 1:
        combined_mesh = trimesh.util.concatenate(combined_meshes)
    else:
        combined_mesh = combined_meshes[0]
    return [combined_mesh], origin


def hussar_3d_terrain_ver2(
    difficulty: float, cfg: "Hussar3DTerrainVer2Cfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a 3D terrain in project hussar.
    """
    difficulty = 0.0
    has_tree = cfg.has_tree
    has_ceil = cfg.has_ceil
    has_maze = cfg.has_maze

    ceil_min_height = cfg.ceil_min_height
    ceil_max_height = cfg.ceil_max_height
    min_gap = cfg.min_gap
    max_gap = cfg.max_gap
    
    # 计算地形位置
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.3)
    size_x, size_y = cfg.size
    
    x0 = [size_x, size_y, 0]
    x1 = [size_x, 0.0, 0]
    x2 = [0.0, size_y, 0]
    x3 = [0.0, 0.0, 0]
    # generate the tri-mesh with two triangles
    vertices = np.array([x0, x1, x2, x3])
    faces = np.array([[1, 0, 2], [2, 3, 1]])
    plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    combined_meshes = [plane_mesh]
    
    # difficulty ∈ [0, 1]
    if has_ceil:
        # 根据难度计算天花板高度
        ceiling_height = ceil_max_height - difficulty * (ceil_max_height - ceil_min_height)
        
        # 创建天花板障碍物
        num_ceiling_parts = int(20 + difficulty * 10)  # 随难度增加天花板数量
        
        for _ in range(num_ceiling_parts):
            # 随机确定天花板大小
            width = random.uniform(1.0, 2.0)
            depth = random.uniform(1.0, 2.0)
            height = random.uniform(0.1, 0.3)
            
            # 随机位置，确保在场景范围内
            x = random.uniform(-size_x/2 + width/2, size_x/2 - width/2)
            y = random.uniform(-size_y/2 + depth/2, size_y/2 - depth/2)
            
            # 高度在计算的范围内随机浮动一点
            z = ceiling_height + random.uniform(0.0, 0.5) + height/2
            
            # 创建天花板块
            ceiling_part = trimesh.creation.box(
                extents=[width, depth, height]
            )
            
            # 应用平移
            translation = trimesh.transformations.translation_matrix([x, y, z])
            ceiling_part.apply_transform(translation)
            
            combined_meshes.append(ceiling_part)
    
    if has_tree:
        # 根据难度计算树干之间的间隙
        current_gap = max_gap - difficulty * (max_gap - min_gap)
        
        # 计算树干的数量和分布
        tree_density = 0.2 + difficulty * 0.4  # 随难度增加树的密度
        area = size_x * size_y
        num_trees = int(area * tree_density / (current_gap * current_gap))
        
        # 限制最大树数量，避免过度密集
        num_trees = min(num_trees, 200)
        
        # 跟踪已放置的树位置
        tree_positions = []
        
        for _ in range(num_trees):
            # 生成候选位置
            attempt = 0
            valid_position = False
            
            while attempt < 50 and not valid_position:  # 最多尝试50次找位置
                x = random.uniform(-size_x/2 + 0.5, size_x/2 - 0.5)
                y = random.uniform(-size_y/2 + 0.5, size_y/2 - 0.5)
                
                # 检查与其他树的距离
                valid_position = True
                for pos in tree_positions:
                    dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                    if dist < current_gap:
                        valid_position = False
                        break
                        
                attempt += 1
            
            if valid_position:
                tree_positions.append((x, y))
                
                # 树干参数
                trunk_height = random.uniform(1.5, 4.0)
                trunk_radius = random.uniform(0.1, 0.3)
                
                # 创建树干
                trunk = trimesh.creation.cylinder(
                    radius=trunk_radius,
                    height=trunk_height,
                    sections=8
                )
                
                # 随机倾斜角度 (0-20度)
                tilt_angle = random.uniform(0, 20) * (np.pi / 180)
                tilt_direction = random.uniform(0, 2 * np.pi)
                
                # 计算倾斜轴和矩阵
                tilt_axis = [np.cos(tilt_direction), np.sin(tilt_direction), 0]
                tilt_matrix = trimesh.transformations.rotation_matrix(tilt_angle, tilt_axis)
                
                # 应用变换
                bottom_to_origin = trimesh.transformations.translation_matrix([0, 0, trunk_height/2])
                translation = trimesh.transformations.translation_matrix([x, y, 0])
                
                trunk.apply_transform(bottom_to_origin)
                trunk.apply_transform(tilt_matrix)
                trunk.apply_transform(translation)
                
                combined_meshes.append(trunk)
    
    if has_maze:
        # 根据难度调整迷宫通道宽度
        passage_width = max_gap - difficulty * (max_gap - min_gap)
        
        # 墙体参数
        wall_height = 2.0
        wall_thickness = 0.3
        
        # 创建随机墙体迷宫，而不是使用DFS算法
        # 生成水平和垂直的墙体，确保它们之间有足够的通道宽度
        
        # 确定墙体数量，随难度增加
        num_walls = int(5 + difficulty * 15)
        
        # 已放置墙体的位置记录
        wall_positions = []
        
        # 墙体的最小和最大长度
        min_wall_length = 2.0
        max_wall_length = min(size_x, size_y) * 0.4
        
        # 生成墙体
        for _ in range(num_walls):
            # 随机决定墙体方向 (水平或垂直)
            is_horizontal = random.choice([True, False])
            
            # 随机墙体长度
            wall_length = random.uniform(min_wall_length, max_wall_length)
            
            # 决定是否在墙体上挖洞（70%的概率挖洞）
            create_hole = random.random() < 0.7
            
            # 墙体尺寸
            if is_horizontal:
                wall_size = [wall_length, wall_thickness, wall_height]
            else:
                wall_size = [wall_thickness, wall_length, wall_height]
            
            # 尝试放置墙体
            max_attempts = 50
            for attempt in range(max_attempts):
                # 随机位置
                x = random.uniform(-size_x/2 + wall_length/2, size_x/2 - wall_length/2) if is_horizontal else random.uniform(-size_x/2 + wall_thickness/2, size_x/2 - wall_thickness/2)
                y = random.uniform(-size_y/2 + wall_thickness/2, size_y/2 - wall_thickness/2) if is_horizontal else random.uniform(-size_y/2 + wall_length/2, size_y/2 - wall_length/2)
                
                # 检查是否与已有墙体太近
                too_close = False
                for wall_pos, wall_is_horizontal, wall_len in wall_positions:
                    # 如果方向相同且平行很近
                    if is_horizontal == wall_is_horizontal:
                        if is_horizontal:
                            # 水平墙体之间的垂直距离
                            if abs(y - wall_pos[1]) < passage_width:
                                # 检查水平重叠
                                min_x = max(x - wall_length/2, wall_pos[0] - wall_len/2)
                                max_x = min(x + wall_length/2, wall_pos[0] + wall_len/2)
                                if max_x > min_x:  # 存在重叠
                                    too_close = True
                                    break
                        else:
                            # 垂直墙体之间的水平距离
                            if abs(x - wall_pos[0]) < passage_width:
                                # 检查垂直重叠
                                min_y = max(y - wall_length/2, wall_pos[1] - wall_len/2)
                                max_y = min(y + wall_length/2, wall_pos[1] + wall_len/2)
                                if max_y > min_y:  # 存在重叠
                                    too_close = True
                                    break
                    # 如果方向不同，检查交叉点附近是否有足够空间
                    else:
                        if is_horizontal:
                            # 水平墙与垂直墙
                            h_x, h_y = x, y
                            v_x, v_y = wall_pos
                            h_len = wall_length
                            v_len = wall_len
                        else:
                            # 垂直墙与水平墙
                            v_x, v_y = x, y
                            h_x, h_y = wall_pos
                            v_len = wall_length
                            h_len = wall_len
                        
                        # 检查是否在彼此范围内
                        if (abs(h_y - v_y) < passage_width and 
                            abs(v_x - h_x) < passage_width and
                            v_x - passage_width < h_x + h_len/2 and 
                            v_x + passage_width > h_x - h_len/2 and
                            h_y - passage_width < v_y + v_len/2 and
                            h_y + passage_width > v_y - v_len/2):
                            too_close = True
                            break
                
                if not too_close:
                    # 找到有效位置，创建墙体
                    
                    # 决定是否挖洞及洞的位置
                    if create_hole and wall_length > passage_width * 2 + 1.0:
                        # 确定洞的位置（避免太靠近墙体边缘）
                        hole_margin = 0.5  # 洞距离墙体边缘的最小距离
                        hole_position = random.uniform(hole_margin, wall_length - passage_width - hole_margin)
                        
                        # 计算墙体左右/上下两段
                        if is_horizontal:
                            # 水平墙体，左段
                            left_length = hole_position
                            left_x = x - wall_length/2 + left_length/2
                            left_y = y
                            left_size = [left_length, wall_thickness, wall_height]
                            
                            # 水平墙体，右段
                            right_length = wall_length - hole_position - passage_width
                            right_x = x + wall_length/2 - right_length/2
                            right_y = y
                            right_size = [right_length, wall_thickness, wall_height]
                            
                            # 创建左段墙体
                            if left_length > 0.2:  # 长度足够才创建
                                left_wall = trimesh.creation.box(extents=left_size)
                                translation = trimesh.transformations.translation_matrix([left_x, left_y, wall_height/2])
                                left_wall.apply_transform(translation)
                                combined_meshes.append(left_wall)
                            
                            # 创建右段墙体
                            if right_length > 0.2:  # 长度足够才创建
                                right_wall = trimesh.creation.box(extents=right_size)
                                translation = trimesh.transformations.translation_matrix([right_x, right_y, wall_height/2])
                                right_wall.apply_transform(translation)
                                combined_meshes.append(right_wall)
                        else:
                            # 垂直墙体，下段
                            bottom_length = hole_position
                            bottom_x = x
                            bottom_y = y - wall_length/2 + bottom_length/2
                            bottom_size = [wall_thickness, bottom_length, wall_height]
                            
                            # 垂直墙体，上段
                            top_length = wall_length - hole_position - passage_width
                            top_x = x
                            top_y = y + wall_length/2 - top_length/2
                            top_size = [wall_thickness, top_length, wall_height]
                            
                            # 创建下段墙体
                            if bottom_length > 0.2:  # 长度足够才创建
                                bottom_wall = trimesh.creation.box(extents=bottom_size)
                                translation = trimesh.transformations.translation_matrix([bottom_x, bottom_y, wall_height/2])
                                bottom_wall.apply_transform(translation)
                                combined_meshes.append(bottom_wall)
                            
                            # 创建上段墙体
                            if top_length > 0.2:  # 长度足够才创建
                                top_wall = trimesh.creation.box(extents=top_size)
                                translation = trimesh.transformations.translation_matrix([top_x, top_y, wall_height/2])
                                top_wall.apply_transform(translation)
                                combined_meshes.append(top_wall)
                    else:
                        # 创建完整墙体（无洞）
                        wall = trimesh.creation.box(extents=wall_size)
                        translation = trimesh.transformations.translation_matrix([x, y, wall_height/2])
                        wall.apply_transform(translation)
                        combined_meshes.append(wall)
                    
                    # 记录墙体位置
                    wall_positions.append(([x, y], is_horizontal, wall_length))
                    break
    
    # 合并所有网格
    if len(combined_meshes) > 1:
        combined_mesh = trimesh.util.concatenate(combined_meshes)
    else:
        combined_mesh = combined_meshes[0]
    
    return [combined_mesh], origin


def hussar_stairs_terrain(
    difficulty: float, cfg: "HussarStairCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    step_width = cfg.step_width_range[1] - difficulty * (cfg.step_width_range[1] - cfg.step_width_range[0])
    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * step_width, terrain_size[1] - 2 * k * step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        # -- dimensions
        box_height = (k + 2) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (step_width, box_size[1], box_height)
        else:
            box_dims = (step_width, box_size[1] - 2 * step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]

    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])

    return meshes_list, origin

def hussar_inverted_stairs_terrain(
    difficulty: float, cfg: "HussarInvertedStairCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a inverted pyramid stair pattern.

    The terrain is an inverted pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    step_width = cfg.step_width_range[1] - difficulty * (cfg.step_width_range[1] - cfg.step_width_range[0])
    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))
    # total height of the terrain
    total_height = (num_steps + 1) * step_height

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders
    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * step_width, terrain_size[1] - 2 * k * step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        # -- dimensions
        box_height = total_height - (k + 1) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (step_width, box_size[1], box_height)
        else:
            box_dims = (step_width, box_size[1] - 2 * step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]
    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])

    return meshes_list, origin

@configclass
class Hussar3DTerrainVer2Cfg(SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = hussar_3d_terrain_ver2

    """whether to add a stair plane to the terrain"""
    has_tree: bool = False
    has_ceil: bool = True
    has_maze: bool = True

    ceil_min_height: float = 0.7
    ceil_max_height: float = 1.4
    min_gap: float = 0.5
    max_gap: float = 2.0

@configclass
class HussarCeilCfg(SubTerrainBaseCfg):
    function = hussar_ceil_terrain
    ceil_min_height: float = 0.9
    ceil_max_height: float = 1.2

@configclass
class HussarPillarCfg(SubTerrainBaseCfg):
    function = hussar_pillars_terrain
    size: tuple[float, float] = (10.0, 10.0)
    max_pillar_radius: float = 0.2
    min_pillar_radius: float = 0.16
    platform_half: float = 1.1
    ground_depth_range: tuple[float, float] = (0.5, 1.0)
    max_gap: float = 1.0
    min_gap: float = 0.6
    full: bool = False # whether to generate full grid

@configclass
class HussarTreeCfg(SubTerrainBaseCfg):
    function = hussar_tree_terrain
    min_gap: float = 1.0
    max_gap: float = 2.0
    
@configclass
class HussarDoorCfg(SubTerrainBaseCfg):
    function = hussar_door_terrain
    min_gap: float = 1.0
    max_gap: float = 2.0

@configclass
class HussarPlatformCfg(SubTerrainBaseCfg):
    function = hussar_platform_terrain
    min_height = 0.3
    max_height = 0.6
    min_gap = 0.3
    max_gap = 0.7
    
@configclass
class HussarStairCfg(SubTerrainBaseCfg):
    function = hussar_stairs_terrain
    border_width: float = 0.0
    step_height_range: tuple[float, float] = MISSING
    step_width_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    platform_height: float = -1.0
    holes: bool = False
    
@configclass
class HussarInvertedStairCfg(SubTerrainBaseCfg):
    function = hussar_inverted_stairs_terrain
    border_width: float = 0.0
    step_height_range: tuple[float, float] = MISSING
    step_width_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    platform_height: float = -1.0
    holes: bool = False

NEW_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=hussar_terrain_generator,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(
            proportion=0.05,
        ),
        "hussar_ceil": HussarCeilCfg(
            proportion=0.15,
            ceil_min_height=1.0,
            ceil_max_height=1.3,  
        ),
        "hussar_tree": HussarTreeCfg(
            proportion=0.1,
            min_gap=0.7,
            max_gap=1.2,
        ),
        "hussar_door": HussarDoorCfg(
            proportion=0.1,
            min_gap=1.5,
            max_gap=2.0,
        ),
        "hussar_platform": HussarPlatformCfg(
            proportion=0.15,
            min_height = 0.05,
            max_height = 0.35,
            min_gap = 0.2,
            max_gap = 0.5,
        ),
        "hussar_pillar": HussarPillarCfg(
            proportion=0.15,
            max_pillar_radius = 0.15,
            min_pillar_radius = 0.15,
            platform_half = 1.0,
            ground_depth_range = (0.55, 0.65),
            max_gap=0.45,
            min_gap=0.35,
            full=False
        ),
        "pyramid_stairs": HussarStairCfg(
            proportion=0.15,
            step_height_range=(0.00, 0.2),
            step_width_range=(0.3, 0.5),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": HussarInvertedStairCfg(
            proportion=0.15,
            step_height_range=(0.00, 0.2),
            step_width_range=(0.3, 0.5),
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

import copy
# for final visualization
VIS_CFG = copy.deepcopy(NEW_ROUGH_TERRAINS_CFG)
VIS_CFG.num_cols = 2
VIS_CFG.num_rows = 4
VIS_CFG.sub_indices = list(range(8))
VIS_CFG.curriculum = False


ROUGH_TERRAIN_BASE_CFG = TerrainImporterCfg(
    class_type=BetterTerrainImporter,
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=None,
    max_init_terrain_level=0,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    # visual_material=sim_utils.MdlFileCfg(
    #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #     project_uvw=True,
    # ),
    debug_vis=False,
)

TERRAIN_USD_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="usd",
    usd_path="/home/btx0424/lab50/benchmark.usda",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    debug_vis=False,
)

registry.register("terrain", "hussar_3d", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=NEW_ROUGH_TERRAINS_CFG))
registry.register("terrain", "hussar_vis", ROUGH_TERRAIN_BASE_CFG.replace(terrain_generator=VIS_CFG))
registry.register("terrain", "hussar_test", TERRAIN_USD_CFG)
