import unittest

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

from jax_sph import partition
from jax_sph.jax_md import space


@jit
def updater(nbrs_old, r_new, **kwargs):
    nbrs_new = nbrs_old.update(r_new, **kwargs)
    return nbrs_new


class BaseTest(unittest.TestCase):
    def body(self, args, backend, num_partitions, verbose=False):
        r = args["r"]
        box_size = args["box_size"]
        cutoff = args["cutoff"]
        mask_self = args["mask_self"]
        target = args["target"]

        if verbose:
            print(f"Start with {backend} backend and {num_partitions} partition(s)")

        N, dim = r.shape
        num_particles_max = r.shape[0]
        displacement_fn, _ = space.periodic(side=box_size)
        neighbor_fn = partition.neighbor_list(
            displacement_fn,
            box_size,
            r_cutoff=cutoff,
            backend=backend,
            dr_threshold=0.0,
            capacity_multiplier=1.25,
            mask_self=mask_self,
            format=partition.NeighborListFormat.Sparse,
            num_particles_max=num_particles_max,
            num_partitions=num_partitions,
            pbc=np.array([True] * dim),
        )

        nbrs = neighbor_fn.allocate(r, num_particles=N)

        if backend == "matscipy":
            nbrs2 = updater(nbrs_old=nbrs, r_new=r, num_particles=N)
        else:
            nbrs2 = updater(nbrs, r)
        mask_real = nbrs.idx[0] < N
        idx_real = nbrs.idx[:, mask_real]

        if verbose:
            print("Idx: \n", nbrs.idx)
            print("Idx_real: \n", idx_real)

        self.assertFalse(nbrs.did_buffer_overflow, "Buffer overflow (allocate)")
        self.assertFalse(nbrs2.did_buffer_overflow, "Buffer overflow (update)")

        self.assertTrue((nbrs.idx == nbrs2.idx).all(), "allocate differes from update")

        self.assertTrue(
            ((nbrs.idx[0] == N) == (nbrs.idx[1] == N)).all(), "One sided edges"
        )

        self_edges_mask = idx_real[0] == idx_real[1]
        if mask_self:
            self.assertEqual(sum(self_edges_mask), 0.0, "Self edges b/n real particles")
        else:
            self_edges = idx_real[:, self_edges_mask]
            self.assertEqual(len(np.unique(self_edges[0])), N, "Self edges are broken")

        # sorted edge list based on second edge row (first sort by first row)
        sort_idx = np.argsort(idx_real[0])
        idx_real_sorted = idx_real[:, sort_idx]
        sort_idx = np.argsort(idx_real_sorted[1])
        idx_real_sorted = idx_real_sorted[:, sort_idx]
        self.assertTrue((idx_real_sorted == target).all(), "Wrong edge list")

        if verbose:
            print(f"Finish with {backend} backend and {num_partitions} partition(s)")

    def cases(self, backend, num_partitions=1, verbose=False):
        # Simple test with pbc and with/without self-masking
        args = {
            "mask_self": False,
            "cutoff": 0.33,
            "box_size": np.array([1.0, 1.0]),
            "r": jnp.array([[0.1, 0.1], [0.1, 0.3], [0.1, 0.9], [0.6, 0.5]]),
            "target": jnp.array([[0, 1, 2, 0, 1, 0, 2, 3], [0, 0, 0, 1, 1, 2, 2, 3]]),
        }
        self.body(args, backend, num_partitions, verbose)

        args["mask_self"] = True
        args["target"] = jnp.array([[1, 2, 0, 0], [0, 0, 1, 2]])
        self.body(args, backend, num_partitions, verbose)

        # Edge case at which the scan implementation almost breaks
        args = {
            "mask_self": False,
            "cutoff": 0.33,
            "box_size": np.array([1.0, 1.0]),
            "r": jnp.array(
                [[0.5, 0.2], [0.2, 0.5], [0.5, 0.5], [0.8, 0.5], [0.5, 0.8]]
            ),
            "target": jnp.array(
                [
                    [0, 2, 1, 2, 0, 1, 2, 3, 4, 2, 3, 2, 4],
                    [0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4],
                ]
            ),
        }
        self.body(args, backend, num_partitions, verbose)

        args["mask_self"] = True
        args["target"] = jnp.array([[2, 2, 0, 1, 3, 4, 2, 2], [0, 1, 2, 2, 2, 2, 3, 4]])
        self.body(args, backend, num_partitions, verbose)

    def test_vmap(self):
        self.cases("jaxmd_vmap")

    def test_scan1(self):
        self.cases("jaxmd_scan")

    def test_scan2(self):
        self.cases("jaxmd_scan", 2)

    def test_matscipy(self):
        self.cases("matscipy")


if __name__ == "__main__":
    unittest.main()
