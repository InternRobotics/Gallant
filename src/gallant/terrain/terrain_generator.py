import inspect
import trimesh
import numpy as np
import os
import torch

from typing import TYPE_CHECKING
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.dict import dict_to_md5_hash
from active_adaptation.envs.terrain import BetterTerrainGenerator

from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg


class hussar_terrain_generator(BetterTerrainGenerator):

    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # randomly sample sub-terrains
        self.sub_terrain_types = torch.zeros(self.cfg.num_rows * self.cfg.num_cols, dtype=torch.int32)
        self.sub_terrain_names = []
        self.terrain_type_names = list(self.cfg.sub_terrains.keys())
        
        self.np_rng = np.random.default_rng(0)
        
        if (sub_indices := self.cfg.sub_indices) is None:
            sub_indices = [
                self.np_rng.choice(len(proportions), p=proportions)
                for _ in range(self.cfg.num_rows * self.cfg.num_cols)
            ]
        else:
            assert len(sub_indices) == self.cfg.num_rows * self.cfg.num_cols, "Number of sub-indices must match the number of terrains"
        
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # randomly sample terrain index
            sub_index = sub_indices[index]
            # randomly sample difficulty parameter
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            # generate terrain
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index], sub_row, sub_col)
            # add to sub-terrains
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])
            self.sub_terrain_types[index] = sub_index
            self.sub_terrain_names.append(self.terrain_type_names[sub_index])
        self.sub_terrain_names = np.array(self.sub_terrain_names)

    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # find the sub-terrain index for each column
        # we generate the terrains based on their proportion (not randomly sampled)
        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # curriculum-based sub-terrains
        self.sub_terrain_types = torch.zeros(self.cfg.num_rows * self.cfg.num_cols, dtype=torch.int32)
        self.sub_terrain_names = []
        self.terrain_type_names = list(self.cfg.sub_terrains.keys())
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                # vary the difficulty parameter linearly over the number of rows
                # note: based on the proportion, multiple columns can have the same sub-terrain type.
                #  Thus to increase the diversity along the rows, we add a small random value to the difficulty.
                #  This ensures that the terrains are not exactly the same. For example, if the
                #  the row index is 2 and the number of rows is 10, the nominal difficulty is 0.2.
                #  We add a small random value to the difficulty to make it between 0.2 and 0.3.
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_indices[sub_col]], sub_row, sub_col)
                # add to sub-terrains
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_indices[sub_col]])
                
                index = sub_row * self.cfg.num_cols + sub_col
                sub_index = sub_indices[sub_col]
                self.sub_terrain_types[index] = sub_index
                self.sub_terrain_names.append(self.terrain_type_names[sub_index])
        self.sub_terrain_names = np.array(self.sub_terrain_names)
                
    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg, sub_row: int, sub_col: int) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Generate a sub-terrain mesh based on the input difficulty parameter.

        If caching is enabled, the sub-terrain is cached and loaded from the cache if it exists.
        The cache is stored in the cache directory specified in the configuration.

        .. Note:
            This function centers the 2D center of the mesh and its specified origin such that the
            2D center becomes :math:`(0, 0)` instead of :math:`(size[0] / 2, size[1] / 2).

        Args:
            difficulty: The difficulty parameter.
            cfg: The configuration of the sub-terrain.

        Returns:
            The sub-terrain mesh and origin.
        """
        # copy the configuration
        cfg = cfg.copy()
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # generate hash for the sub-terrain
        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        # generate the file name
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")

        # check if hash exists - if true, load the mesh and origin and return
        if self.cfg.use_cache and os.path.exists(sub_terrain_obj_filename):
            # load existing mesh
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            # return the generated mesh
            return mesh, origin

        # generate the terrain
        if 'num_col' in inspect.signature(cfg.function).parameters.keys():
            meshes, origin = cfg.function(difficulty, cfg, num_row=sub_row, num_col=sub_col)
        else:
            meshes, origin = cfg.function(difficulty, cfg)
        mesh = trimesh.util.concatenate(meshes)
        # offset mesh such that they are in their center
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # change origin to be in the center of the sub-terrain
        origin += transform[0:3, -1]

        # if caching is enabled, save the mesh and origin
        if self.cfg.use_cache:
            # create the cache directory
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            # save the data
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)
        # return the generated mesh
        return mesh, origin