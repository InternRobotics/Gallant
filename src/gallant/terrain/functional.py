import trimesh
import numpy as np
from shapely.geometry import box as shapely_box


def hussar_pillars_terrain(
    difficulty: float, cfg: "HussarPillarCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    场景结构（改后逻辑）：
      - 先按六角密堆把"所有该有的桩"都生成（不因平台而删桩）
      - 再把平台作为"盖板"挤出，盖在 z=0 之上（稍微抬一点点避免与桩顶共面）
      - 外围空间：挖到 -ground_depth，并保留一块底面（base plate）+ 周边立墙（skirt）

    参数：
      - difficulty: 难度系数 (0.0-1.0)
      - cfg: 配置对象
      - full: 是否生成完整网格，False时只在中心十字形区域生成桩

    需要 cfg：
      - size: (size_x, size_y)
      - min_gap, max_gap
    可选 cfg：
      - max_pillar_radius, min_pillar_radius 或 pillar_radius（择一）
      - ground_depth (默认 2.0)
      - base_thickness (默认 0.2)
      - platform_half (默认 1.1)
      - platform_thickness_top (默认 0.02)            # 盖板厚度，向 +z 挤出
      - platform_epsilon (默认 1e-4)                   # 将平台整体抬起一点避免共面
      - skirt_thickness (默认 0.15)                    # 立墙环宽
      - skirt_depth (默认 = ground_depth)              # 立墙深度（从 0 到 -skirt_depth）
      - pillar_clearance_between (默认 0.05)
    """
    # ---------------- 基本参数 ----------------
    size_x, size_y = cfg.size
    d_raw = float(np.clip(difficulty, 0.0, 1.0))

    # 分段难度映射：
    # - d<=0.6: 分两段独立调参 [0.1,0.3] 与 [0.4,0.6]
    # - d>0.6 : 单段 [0.6,1.0]
    if d_raw <= 0.6:
        if d_raw <= 0.3:
            alpha = (d_raw) / 0.3   # 0..1 映射
        elif d_raw <= 0.6:
            alpha = (d_raw - 0.3) / 0.3  # 0..1 映射
    else:
        alpha = (d_raw - 0.6) / 0.4  # 0..1 映射

    # 间距与半径
    spacing_target = cfg.min_gap + alpha * (cfg.max_gap - cfg.min_gap)

    if hasattr(cfg, "max_pillar_radius") and hasattr(cfg, "min_pillar_radius"):
        pillar_radius = cfg.max_pillar_radius - alpha * (
            cfg.max_pillar_radius - cfg.min_pillar_radius
        )
    else:
        pillar_radius = getattr(cfg, "pillar_radius", 0.25)

    ground_depth_range = getattr(cfg, "ground_depth_range", (1.0, 2.0))
    ground_depth = np.random.uniform(ground_depth_range[0], ground_depth_range[1])
    
    base_thickness = getattr(cfg, "base_thickness", 0.2)
    platform_half = getattr(cfg, "platform_half", 0.25)
    pillar_clearance_between = getattr(cfg, "pillar_clearance_between", 0.05)

    # 盖板参数：平台盖在 z=0 之上
    platform_thickness_top = getattr(cfg, "platform_thickness_top", 0.02)
    # platform_epsilon = getattr(cfg, "platform_epsilon", 1e-4)
    platform_epsilon = -platform_thickness_top

    # 立墙参数
    skirt_thickness = getattr(cfg, "skirt_thickness", 0.15)
    skirt_depth = getattr(cfg, "skirt_depth", ground_depth)

    # 最小间距限制避免桩互相接触
    min_spacing = 2.0 * pillar_radius + pillar_clearance_between
    spacing = max(spacing_target, min_spacing)

    # 六角密堆步长
    dx = spacing
    dy = spacing * np.sqrt(3.0) / 2.0

    # 安全边界，确保桩完全在地图内
    margin_x = pillar_radius + 1e-4
    margin_y = pillar_radius + 1e-4

    # 参考点（机器人生成点等）
    cx, cy = size_x / 2.0, size_y / 2.0
    origin = np.array([cx, cy, 0.3])

    combined = []

    # ---------------- 底板（整块） ----------------
    # 位于 [-ground_depth, -ground_depth + base_thickness]
    full_rect = shapely_box(0.0, 0.0, size_x, size_y)
    base_plate = trimesh.creation.extrude_polygon(full_rect, height=base_thickness)
    base_plate.apply_transform(trimesh.transformations.translation_matrix([0, 0, -ground_depth]))
    combined.append(base_plate)

    # ---------------- 周边立墙（裙边） ----------------
    outer = shapely_box(0.0, 0.0, size_x, size_y)
    inner = shapely_box(skirt_thickness, skirt_thickness, size_x - skirt_thickness, size_y - skirt_thickness)
    skirt_ring = outer.difference(inner)
    perimeter_skirt = trimesh.creation.extrude_polygon(skirt_ring, height=skirt_depth)  # 从 0 向 +z
    perimeter_skirt.apply_transform(trimesh.transformations.translation_matrix([0, 0, -skirt_depth]))  # 下移
    combined.append(perimeter_skirt)

    # ---------------- 全域铺设桩（不因平台而删） ----------------
    pillar_height = ground_depth            # 顶到 z=0，底到 z=-ground_depth
    z_center = -ground_depth / 2.0          # 圆柱中心 z

    y = margin_y
    center_x, center_y = size_x / 2.0, size_y / 2.0
    # 根据难度决定铺设方式
    full_all = d_raw <= 0.6
    # 对于 d>0.6，使用多条线从4逐渐减少到2
    if not full_all:
        t_line = (d_raw - 0.6) / 0.4
        t_line = float(np.clip(t_line, 0.0, 1.0))
        num_lines = int(round(4 - 2 * t_line))
        num_lines = max(2, min(4, num_lines))
        # 仅在中心附近布置若干条水平/垂直线：以中心为对称，线间距≈spacing
        cluster_half_y = spacing * (num_lines - 1) * 0.5
        cluster_half_x = spacing * (num_lines - 1) * 0.5
        ys = center_y + np.linspace(-cluster_half_y, cluster_half_y, num=num_lines)
        xs = center_x + np.linspace(-cluster_half_x, cluster_half_x, num=num_lines)
        # 保证在安全边界内
        ys = np.clip(ys, margin_y, size_y - margin_y)
        xs = np.clip(xs, margin_x, size_x - margin_x)
        line_tolerance = spacing * 0.6
    row_idx = 0
    while y <= size_y:
        x_start = margin_x + (dx / 2.0 if (row_idx % 2 == 1) else 0.0)
        x = x_start
        while x <= size_x:
            # Check if we should place a pillar at this position
            should_place = True
            if not full_all:
                # 仅在若干条水平/垂直线上放置桩
                on_horizontal = np.any(np.abs(y - ys) < line_tolerance)
                on_vertical = np.any(np.abs(x - xs) < line_tolerance)
                should_place = bool(on_horizontal or on_vertical)
            
            if should_place:
                cyl = trimesh.creation.cylinder(radius=pillar_radius, height=pillar_height, sections=24)
                cyl.apply_transform(trimesh.transformations.translation_matrix([x, y, z_center]))
                combined.append(cyl)
            x += dx
        y += dy
        row_idx += 1

    # ---------------- 平台盖板（向 +z 挤出，盖在桩顶之上） ----------------
    plat_xmin, plat_xmax = cx - platform_half, cx + platform_half
    plat_ymin, plat_ymax = cy - platform_half, cy + platform_half

    platform_rect = shapely_box(plat_xmin, plat_ymin, plat_xmax, plat_ymax)
    center_plate = trimesh.creation.extrude_polygon(platform_rect, height=platform_thickness_top)  # 0 → +厚度
    # 为避免与桩顶共面、渲染抖动，整体抬升一个很小的 epsilon
    center_plate.apply_transform(trimesh.transformations.translation_matrix([0, 0, platform_epsilon]))
    combined.append(center_plate)

    # ---------------- 合并并返回 ----------------
    return [trimesh.util.concatenate(combined)], origin