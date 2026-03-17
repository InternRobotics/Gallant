import active_adaptation
from pathlib import Path

if active_adaptation.get_backend() == "isaac":
    from . import hussar_terrain
else:
    from active_adaptation.envs.terrain import TERRAINS_MUJOCO, MjTerrainCfg
    path = Path(__file__).parent / "ground.xml"
    TERRAINS_MUJOCO["hussar_3d"] = MjTerrainCfg(mjcf_path=str(path))