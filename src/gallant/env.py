"""Gallant Isaac Lab env: stock backend + pillar collision skins."""
from __future__ import annotations

from typing_extensions import override

from active_adaptation.envs import IsaacBackendEnv


def add_skin_by_tiles(
    terrain_cfg,
    z_plane: float = 0.0,
    thickness: float = 0.02,
    prim_root: str = "/World/terrain_skins",
    col_axis: str = "x",
) -> None:
    """Spawn invisible kinematic skins over hussar_pillar tiles (collision only)."""
    from isaaclab.sim.spawners.shapes import spawn_cuboid, CuboidCfg
    from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg

    gen = terrain_cfg.terrain_generator
    if gen is None or "hussar_pillar" not in getattr(gen, "sub_terrains", {}):
        return

    num_rows = int(gen.num_rows)
    num_cols = int(gen.num_cols)
    tile_sx, tile_sy = gen.sub_terrains["hussar_pillar"].size
    half_h = thickness * 0.5

    keys = list(gen.sub_terrains.keys())
    props = [float(gen.sub_terrains[k].proportion) for k in keys]
    s = sum(props) or 1.0
    props = [p / s for p in props]

    boundaries = [0]
    cum = 0.0
    for p in props[:-1]:
        cum += p
        boundaries.append(int(round(cum * num_cols)))
    boundaries.append(num_cols)
    key_cols = {k: (boundaries[i], boundaries[i + 1]) for i, k in enumerate(keys)}
    c0, c1 = key_cols["hussar_pillar"]
    target_rows = range(0, 3)
    target_cols = range(c0, c1)

    x0 = -0.5 * num_rows * tile_sy
    y0 = -0.5 * num_cols * tile_sx

    skin_cfg = CuboidCfg(
        size=(tile_sx, tile_sy, thickness),
        visible=False,
        rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=CollisionPropertiesCfg(
            collision_enabled=True, rest_offset=0.0, contact_offset=0.01
        ),
    )

    placed = 0
    for r in target_rows:
        for c in target_cols:
            if col_axis == "x":
                cx = x0 + (c + 0.5) * tile_sx
                cy = y0 + (r + 0.5) * tile_sy
            else:
                cx = x0 + (r + 0.5) * tile_sx
                cy = y0 + (c + 0.5) * tile_sy
            cz = float(z_plane) + half_h
            prim_path = f"{prim_root}/pillar_skin_r{r:02d}_c{c:02d}"
            spawn_cuboid(prim_path=prim_path, cfg=skin_cfg, translation=(cx, cy, cz))
            placed += 1

    print(
        f"[skin-by-tiles] covered hussar_pillar cols {c0}-{c1 - 1}, "
        f"rows 0-2, placed {placed}. origin=({x0:.3f},{y0:.3f}), "
        f"z_plane={z_plane:.3f}, axis={col_axis}"
    )


class GallantEnvIsaac(IsaacBackendEnv):
    """Isaac Lab backend with Gallant pillar collision skins after scene build."""

    supported_backends = ("isaaclab",)

    # @override
    # def setup_scene(self):
    #     super().setup_scene()
    #     terrain = getattr(self.scene, "terrain", None)
    #     cfg = getattr(terrain, "cfg", None) if terrain is not None else None
    #     if cfg is None:
    #         return
    #     from isaaclab.sim import use_stage

        # with use_stage(self.sim._sim.get_initial_stage()):
        #     add_skin_by_tiles(cfg, z_plane=-0.02, thickness=0.02, col_axis="y")
