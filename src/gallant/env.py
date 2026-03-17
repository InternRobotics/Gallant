import active_adaptation
from active_adaptation.envs import IsaacBackendEnv
from active_adaptation.registry import Registry
from active_adaptation.assets import AssetCfg
from active_adaptation.envs.backends.isaac import (
    IsaacSceneAdapter, IsaacSimAdapter,
)

if active_adaptation.get_backend() == "isaac":
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim.spawners.shapes import spawn_cuboid, CuboidCfg
    from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg

    def add_skin_by_tiles(
        scene_cfg: InteractiveSceneCfg,
        z_plane=0.0,              # 想贴合的全局物理高度
        thickness=0.02,           # 薄板厚度
        prim_root="/World/terrain_skins",
        col_axis="x",             # 'x' 表示列沿 X 方向，'y' 表示列沿 Y
    ):
        gen = scene_cfg.terrain.terrain_generator
        num_rows = int(getattr(gen, "num_rows"))
        num_cols = int(getattr(gen, "num_cols"))

        # 单块尺寸——以 pillar 的 size 为准（你这套是固定 8×8）
        tile_sx, tile_sy = gen.sub_terrains["hussar_pillar"].size
        half_h = thickness * 0.5

        # 1) 按 key 顺序 + proportion 划分列范围
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
        key_cols = {k: (boundaries[i], boundaries[i+1]) for i, k in enumerate(keys)}
        c0, c1 = key_cols["hussar_pillar"]                 # [c0, c1) 是 pillar 的列段
        target_rows = range(0, 3)              # 前半行
        target_cols = range(c0, c1)

        # 2) 推断地形网格左下角 (x0, y0) 的“块左下角坐标”（不是中心）
            # 用当前 env 起点的最小 x,y 当作“左下角块中心”，退半块得到左下角角点
            # origins = np.asarray(scene.env_origins.cpu().detach(), dtype=float) if getattr(scene, "env_origins", None) is not None else np.zeros((1,3))
        x0 = -0.5 * num_rows * tile_sy#-0.75 * num_cols * tile_sx
        y0 = -0.5 * num_cols * tile_sx
            # if col_axis == "x":
            #     x0 = origins[:, 0].min() - 0.5 * tile_sx
            #     y0 = origins[:, 1].min() - 0.5 * tile_sy
            # else:
            #     # 列沿 Y 时，仍然用同样的“最小中心退半块”法
            #     x0 = origins[:, 0].min() - 0.5 * tile_sx
            #     y0 = origins[:, 1].min() - 0.5 * tile_sy


        # 3) 生成不可见可碰撞的薄板 cfg
        skin_cfg = CuboidCfg(
            size=(tile_sx, tile_sy, thickness),
            visible=False,
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),   # 静态蒙皮
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True, rest_offset=0.0, contact_offset=0.01
            ),
        )

        # 4) 按 “列∈pillar段 且 行在前半” 的格子铺板（不看 num_envs）
        placed = 0
        for r in target_rows:
            for c in target_cols:
                # 块中心坐标
                if col_axis == "x":
                    cx = x0 + (c + 0.5) * tile_sx
                    cy = y0 + (r + 0.5) * tile_sy
                else:  # 列沿 Y
                    cx = x0 + (r + 0.5) * tile_sx
                    cy = y0 + (c + 0.5) * tile_sy
                cz = float(z_plane) + half_h

                prim_path = f"{prim_root}/pillar_skin_r{r:02d}_c{c:02d}"
                spawn_cuboid(prim_path=prim_path, cfg=skin_cfg, translation=(cx, cy, cz))
                placed += 1

        print(f"[skin-by-tiles] ✓ 覆盖 hussar_pillar 列 {c0}-{c1-1}，行 0-{(num_rows//2)-1}，共放置 {placed} 块。"
            f" 原点(x0,y0)=({x0:.3f},{y0:.3f})，z_plane={z_plane:.3f}, axis={col_axis}")


class GallantEnvIsaac(IsaacBackendEnv):
    """Isaac Sim backend implementation."""
    
    def setup_scene(self):
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationContext, attach_stage_to_usd_context, use_stage
        # from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
        from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
        from isaaclab.assets import AssetBaseCfg, ArticulationCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        
        registry = Registry.instance()
        
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.cfg.num_envs,
            env_spacing=2.5,
            replicate_physics=True
        )
        scene_cfg.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )
        asset_cfg = registry.get("asset", self.cfg.robot.name)
        if isinstance(asset_cfg, AssetCfg):
            scene_cfg.robot = asset_cfg.isaaclab()
            for sensor_cfg in asset_cfg.sensors_isaaclab:
                setattr(scene_cfg, sensor_cfg.name, sensor_cfg.isaaclab())
        elif isinstance(asset_cfg, ArticulationCfg):
            scene_cfg.robot = asset_cfg
            scene_cfg.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                track_air_time=True,
                history_length=3
            )
        else:
            raise ValueError(f"Asset configuration must be an instance of AssetCfg or ArticulationCfg, got {type(asset_cfg)}")
        
        scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        scene_cfg.terrain = registry.get("terrain", self.cfg.terrain)

        sim_cfg = sim_utils.SimulationCfg(
            dt=self.cfg.sim.isaac_physics_dt,
            render=sim_utils.RenderCfg(
                rendering_mode="balanced",
                # antialiasing_mode="FXAA",
                # enable_global_illumination=True,
                # enable_reflections=True,
            ),
            physx=sim_utils.PhysxCfg(
                **self.cfg.sim.get("physx", {}),
            ),
            device=str(self.device)
        )

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(sim_cfg)
        else:
            self.sim: SimulationContext = SimulationContext.instance()
        
        with use_stage(self.sim.get_initial_stage()):
            self.scene = InteractiveScene(scene_cfg)
            add_skin_by_tiles(scene_cfg, z_plane=-0.02, thickness=0.02, col_axis='y')
            attach_stage_to_usd_context()
        
        # TODO@btx0424: check if we need to perform startup randomizations before resetting 
        # the simulation.
        with use_stage(self.sim.get_initial_stage()):
            self.sim.reset()
        
        # set camera view for "/OmniverseKit_Persp" camera
        self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
        try:
            import omni.replicator.core as rep
            # create render product
            self._render_product = rep.create.render_product(
                "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
            )
            # create rgb annotator -- used to read data from the render product
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
        except ModuleNotFoundError as e:
            print("Set enable_cameras=true to use cameras.")            

        self.sim = IsaacSimAdapter(self.sim)
        self.scene = IsaacSceneAdapter(self.scene)
        self.terrain_type = self.scene.terrain.cfg.terrain_type