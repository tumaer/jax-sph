"""Create grid for moving least squares interpolation"""

import numpy as np
from abc import ABC
from jax_sph.utils import pos_init_cartesian_2d, pos_init_cartesian_3d


class interpolationGrid():
        
    def __init__(self, args):

        self.args = args
        self.nx, self.ny, self.nz = [int(i) for i in args.nxnynz.split("_")]
        # length, heigth, and width of fluid domain; box with 3dx not incl.
        dx = args.dx
        self.L, self.H, self.W = self.nx * dx, self.ny * dx, self.nz * dx
   
    def initialize_grid(self):
        args = self.args

        # initialize box and regular grid positions of MLS interpolation grid
        if args.dim == 2:
            box_size = self._box_size2D()
            r = self._init_pos2D(box_size, args.dx)
        elif args.dim == 3:
            box_size = self._box_size3D()
            r = self._init_pos3D(box_size, args.dx)

        num_interp_points = len(r)
        print("Total number of interpolation grid ponts = ", num_interp_points)

        return r, box_size

    def _box_size2D(self):
        wall = 0 #if self.relax_pbc else 6
        return (np.array([self.args.bounds[0][1], self.args.bounds[1][1]]) + wall)

    def _box_size3D(self):
        wall = 0 #if self.relax_pbc else 6
        return (np.array([self.args.bounds[0][1], self.args.bounds[1][1], self.args.bounds[2][1]]) + wall)
    
    def _init_pos2D(self, box_size, dx):
        return pos_init_cartesian_2d(box_size, dx)

    def _init_pos3D(self, box_size, dx):
        return pos_init_cartesian_3d(box_size, dx)