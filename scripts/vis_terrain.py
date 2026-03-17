import trimesh
import active_adaptation
active_adaptation.set_backend("mujoco")

from isaaclab.utils import configclass
from terrain.functional import hussar_pillars_terrain

@configclass
class HussarPillarCfg:
    function = hussar_pillars_terrain
    size: tuple[float, float] = (10.0, 10.0)
    max_pillar_radius: float = 0.2
    min_pillar_radius: float = 0.16
    platform_half: float = 1.1
    ground_depth_range: tuple[float, float] = (0.5, 1.0)
    max_gap: float = 1.0
    min_gap: float = 0.6
    full: bool = False # whether to generate full grid


def main():
    cfg = HussarPillarCfg()
    meshes, origin = hussar_pillars_terrain(0.5, cfg)
    scene = trimesh.Scene(meshes)
    scene.show()

if __name__ == "__main__":
    main()