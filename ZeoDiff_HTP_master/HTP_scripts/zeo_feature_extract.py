# =============================================================================
# Authors: Ji Qi, Xiaobao Wang
# Description:
#   Extracts diffusion-related channel descriptors from zeolite CIF files.
# =============================================================================


import os
import time
import warnings
import numpy as np
from ase.io import read, write
from ase.geometry import find_mic
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
from scipy.ndimage import label

# Zeolites symmetry processing modules
try:
    from pymatgen.core.structure import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    USE_PYMATGEN = True
except ImportError:
    USE_PYMATGEN = False

try:
    import pandas as pd

    USE_PANDAS = True
except ImportError:
    USE_PANDAS = False

# Diffusion-channel clustering analysis
try:
    from sklearn.cluster import DBSCAN

    USE_SKLEARN = True
except ImportError:
    USE_SKLEARN = False

# Visualization module
try:
    import matplotlib.pyplot as plt

    USE_MATPLOTLIB = True
except ImportError:
    USE_MATPLOTLIB = False

warnings.filterwarnings('ignore')

# Optional acceleration backends
USE_CUPY = False
USE_NUMBA = False
try:
    import cupy as cp

    USE_CUPY = True
except Exception:
    USE_CUPY = False

try:
    from numba import njit, prange

    USE_NUMBA = True
except Exception:
    USE_NUMBA = False

# -------------------------------------------------------------------
# Numba-accelerated Lennard-Jones energy evaluation for probe sampling
# -------------------------------------------------------------------
if USE_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _compute_energies_numba(frac_points, frac_atoms, sigma_atoms, eps_atoms,
                                sigma_probe, eps_probe, cell, cutoff, min_dist):
        n_points = frac_points.shape[0]
        n_atoms = frac_atoms.shape[0]
        energies = np.zeros(n_points)
        for ip in prange(n_points):
            fx = frac_points[ip, 0]
            fy = frac_points[ip, 1]
            fz = frac_points[ip, 2]
            total_energy = 0.0
            for ia in range(n_atoms):
                # Minimum-image distance
                dfx = frac_atoms[ia, 0] - fx
                dfy = frac_atoms[ia, 1] - fy
                dfz = frac_atoms[ia, 2] - fz
                dfx = dfx - np.round(dfx)
                dfy = dfy - np.round(dfy)

                # Fractional to Cartesian conversion
                drx = dfx * cell[0, 0] + dfy * cell[1, 0] + dfz * cell[2, 0]
                dry = dfx * cell[0, 1] + dfy * cell[1, 1] + dfz * cell[2, 1]
                drz = dfx * cell[0, 2] + dfy * cell[1, 2] + dfz * cell[2, 2]
                r = np.sqrt(drx * drx + dry * dry + drz * drz)

                # Hard-sphere repulsion
                if r < min_dist:
                    total_energy = 1000.0
                    break

                # LJ potential
                if r <= cutoff and r >= min_dist:
                    sigma_mix = 0.5 * (sigma_atoms[ia] + sigma_probe)
                    eps_mix = np.sqrt(eps_atoms[ia] * eps_probe)
                    sr6 = (sigma_mix / r) ** 6
                    total_energy += 4.0 * eps_mix * (sr6 * (sr6 - 1.0))
            if total_energy > 1000.0:
                total_energy = 1000.0
            energies[ip] = total_energy
        return energies
else:
    def _compute_energies_numba(*args, **kwargs):
        raise RuntimeError("numba not available")


class SymmetryRemover:
    def __init__(self):
        if not USE_PYMATGEN:
            raise ImportError("pymatgen not installed, cannot perform symmetry processing")

    def manual_expand_symmetry_keep_cell(self, structure):
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        sym_ops = sga.get_space_group_operations()
        expanded_sites = []
        for site in structure:
            for sym_op in sym_ops:
                new_coords = sym_op.operate(site.frac_coords)
                new_coords = new_coords % 1.0
                is_duplicate = False
                for existing_site in expanded_sites:
                    if (existing_site['species'] == site.species_string and
                            all(abs(new_coords - existing_site['coords']) < 1e-3)):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    expanded_sites.append({
                        'species': site.species_string,
                        'coords': new_coords
                    })

        expanded_structure = Structure(
            lattice=structure.lattice,
            species=[site['species'] for site in expanded_sites],
            coords=[site['coords'] for site in expanded_sites]
        )

        return expanded_structure

    def process_cif_file(self, input_cif_path, output_cif_path=None):
        try:
            print(f"Processing symmetry: {os.path.basename(input_cif_path)}")

            # Read original structure
            structure = Structure.from_file(input_cif_path)
            original_volume = structure.lattice.volume
            original_atoms = len(structure)

            # Get space group information
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            space_group = sga.get_space_group_symbol()

            # Skip if structure already has no symmetry
            if space_group.strip().upper() == 'P1' or space_group.strip() == 'P 1':
                if output_cif_path and output_cif_path != input_cif_path:
                    import shutil
                    shutil.copy2(input_cif_path, output_cif_path)
                    return output_cif_path, True
                return input_cif_path, True
            # Symmetry expansion
            expanded_structure = self.manual_expand_symmetry_keep_cell(structure)

            final_volume = expanded_structure.lattice.volume
            final_atoms = len(expanded_structure)
            volume_change = abs(final_volume - original_volume) / original_volume

            if volume_change > 1e-6:
                print(f"Warning: Minor change in cell volume ({volume_change:.2e})")
            else:
                print(f"Cell parameters remain unchanged")

            # Determine output path
            if output_cif_path is None:
                output_cif_path = input_cif_path

            expanded_structure.to(filename=output_cif_path)
            return output_cif_path, True

        except Exception as e:
            return input_cif_path, False


class ZeolitePotentialGrid:
    """Potential energy grid calculation for zeolites"""

    def __init__(self, cif_file, output_prefix=None, spacing_A=0.2, cutoff_A=14.0, probe_type="ch4"):
        self.cif_file = cif_file
        self.spacing_A = spacing_A
        self.cutoff_A = cutoff_A
        self.probe_type = probe_type.lower()

        if output_prefix is None:
            base_name = os.path.splitext(os.path.basename(cif_file))[0]
            self.output_prefix = os.path.join(os.path.dirname(cif_file), base_name)
        else:
            self.output_prefix = output_prefix

        self.out_csv = f"{self.output_prefix}_energy_grid.csv"
        self.out_vtk = f"{self.output_prefix}_energy_grid"

        # Lennard-Jones parameters (sigma/nm, epsilon/kJ mol^-1)
        self.ff_params = {
            "o": (0.3304, 0.442329411),    # TraPPE-zeo
            "si": (0.2310, 0.18458107),    # TraPPE-zeo
            "ch4": (0.3730, 1.230540467),  # TraPPE-UA
        }

        if self.probe_type not in self.ff_params:
            available = list(self.ff_params.keys())
            raise ValueError(f"Unsupported probe type '{probe_type}'. Available types: {available}")

    def _symbol_to_ff_key(self, symbol):
        clean_symbol = ''.join([c for c in symbol if c.isalpha()]).lower()
        if clean_symbol.startswith("si"):
            return "si"
        elif clean_symbol.startswith("al"):   # Al atoms use Si parameters
            return "si"
        elif clean_symbol.startswith("o"):
            return "o"
        elif clean_symbol in self.ff_params:
            return clean_symbol
        else:
            # Fallback: use oxygen parameters for unknown species
            print(f"Warning: Unrecognized atom symbol '{symbol}'. Using oxygen force field parameters.")
            return "o"

    def expand_cif(self, min_length=28.0):
        """Supercell: ensure minimum cell edge length >= min_length"""
        atoms = read(self.cif_file)
        cell_lengths = atoms.get_cell_lengths_and_angles()[:3]
        print(f"Original cell edge lengths: a={cell_lengths[0]:.3f}, b={cell_lengths[1]:.3f}, c={cell_lengths[2]:.3f} A")

        scale_factors = [int(np.ceil(min_length / L)) if L < min_length else 1 for L in cell_lengths]
        print(f"Recommended expansion factors: {scale_factors}")

        supercell = atoms * scale_factors
        new_lengths = supercell.get_cell_lengths_and_angles()[:3]
        print(f"Cell edge lengths after expansion: a={new_lengths[0]:.3f}, b={new_lengths[1]:.3f}, c={new_lengths[2]:.3f} A")

        base, ext = os.path.splitext(self.cif_file)
        self.cif_file = f"{base}_supercell.cif"
        write(self.cif_file, supercell)
        print(f"Expanded CIF file saved: {self.cif_file}")

    def load_structure(self):
        print(f"Reading CIF file: {self.cif_file}")
        try:
            self.atoms = read(self.cif_file)
        except Exception as e:
            raise RuntimeError(f"Failed to read CIF file: {e}")

        self.cell_A = self.atoms.get_cell()
        self.positions_A = self.atoms.get_positions()
        self.symbols = self.atoms.get_chemical_symbols()

        # Unit conversion: Angstrom to nm
        self.cell = self.cell_A / 10.0
        self.positions = self.positions_A / 10.0
        self.spacing = self.spacing_A / 10.0
        self.cutoff = self.cutoff_A / 10.0

        try:
            self.cell_inv = np.linalg.inv(self.cell)
        except np.linalg.LinAlgError:
            raise RuntimeError("Cell matrix not invertible. Please check CIF cell parameters.")

        self._prepare_atom_parameters()

        print(f"Successfully read {len(self.symbols)} atoms")
        cell_lengths = np.linalg.norm(self.cell, axis=1) * 10.0
        print(f"Cell parameters: a={cell_lengths[0]:.3f} A, b={cell_lengths[1]:.3f} A, c={cell_lengths[2]:.3f} A")
        print(f"Cell volume: {np.linalg.det(self.cell) * 1000:.3f} A^3")

    def _prepare_atom_parameters(self):
        n_atoms = len(self.symbols)
        self.sigma_atoms = np.zeros(n_atoms)
        self.eps_atoms = np.zeros(n_atoms)
        self.frac_atoms = np.zeros((n_atoms, 3))

        self.sigma_probe, self.eps_probe = self.ff_params[self.probe_type]

        for i, (pos, symbol) in enumerate(zip(self.positions, self.symbols)):
            try:
                ff_key = self._symbol_to_ff_key(symbol)
                self.sigma_atoms[i], self.eps_atoms[i] = self.ff_params[ff_key]
                self.frac_atoms[i] = pos @ self.cell_inv
            except KeyError as e:
                print(f"Warning: {e}")
                self.sigma_atoms[i], self.eps_atoms[i] = self.ff_params["o"]
                self.frac_atoms[i] = pos @ self.cell_inv

    def generate_grid(self):
        print("Generating grid...")
        cell_lengths = np.linalg.norm(self.cell, axis=1)
        self.n_grid = [max(1, int(np.ceil(length / self.spacing))) for length in cell_lengths]
        self.n1, self.n2, self.n3 = self.n_grid

        print(f"Grid dimensions: {self.n1} x {self.n2} x {self.n3} = {np.prod(self.n_grid)} grid points")
        print(f"Grid spacing: {self.spacing_A:.2f} A")

        self.frac_x = np.linspace(0.0, 1.0, self.n1, endpoint=False)
        self.frac_y = np.linspace(0.0, 1.0, self.n2, endpoint=False)
        self.frac_z = np.linspace(0.0, 1.0, self.n3, endpoint=False)

        self.grid_energies = np.zeros((self.n1, self.n2, self.n3))

    # ---------- Numba parallel functions (if available) --------------
    if USE_NUMBA:
        @staticmethod
        @njit(parallel=True, fastmath=True)
        def _compute_energies_numba(frac_points, frac_atoms, sigma_atoms, eps_atoms,
                                    sigma_probe, eps_probe, cell, cutoff, min_dist):
            n_points = frac_points.shape[0]
            n_atoms = frac_atoms.shape[0]
            energies = np.zeros(n_points)
            for ip in prange(n_points):
                fx = frac_points[ip, 0]
                fy = frac_points[ip, 1]
                fz = frac_points[ip, 2]
                total_energy = 0.0
                for ia in range(n_atoms):
                    dfx = frac_atoms[ia, 0] - fx
                    dfy = frac_atoms[ia, 1] - fy
                    dfz = frac_atoms[ia, 2] - fz
                    dfx = dfx - np.round(dfx)
                    dfy = dfy - np.round(dfy)
                    dfz = dfz - np.round(dfz)
                    drx = dfx * cell[0, 0] + dfy * cell[1, 0] + dfz * cell[2, 0]
                    dry = dfx * cell[0, 1] + dfy * cell[1, 1] + dfz * cell[2, 1]
                    drz = dfx * cell[0, 2] + dfy * cell[1, 2] + dfz * cell[2, 2]
                    r = np.sqrt(drx * drx + dry * dry + drz * drz)
                    if r < min_dist:
                        total_energy = 1000.0
                        break
                    if r <= cutoff and r >= min_dist:
                        sigma_mix = 0.5 * (sigma_atoms[ia] + sigma_probe)
                        eps_mix = np.sqrt(eps_atoms[ia] * eps_probe)
                        sr6 = (sigma_mix / r) ** 6
                        total_energy += 4.0 * eps_mix * (sr6 * (sr6 - 1.0))
                if total_energy > 1000.0:
                    total_energy = 1000.0
                energies[ip] = total_energy
            return energies

    def calculate_potential(self):
        """Compute probe-framework potential energy on the fractional grid"""
        print(f"Computing {self.probe_type.upper()} probe potential energy distribution...")
        print(f"Cutoff distance: {self.cutoff_A:.1f} A")

        total_points = int(np.prod(self.n_grid))
        min_distance_A = 0.1
        min_distance_nm = min_distance_A / 10.0

        # Generate all fractional coordinate points (N,3)
        FX, FY, FZ = np.meshgrid(self.frac_x, self.frac_y, self.frac_z, indexing='ij')
        frac_points = np.vstack([FX.ravel(), FY.ravel(), FZ.ravel()]).T

        # Prepare atom array (numpy)
        frac_atoms = self.frac_atoms.copy()
        sigma_atoms = self.sigma_atoms.copy()
        eps_atoms = self.eps_atoms.copy()
        sigma_probe = self.sigma_probe
        eps_probe = self.eps_probe
        cell = self.cell.copy()
        cutoff = self.cutoff
        min_dist = min_distance_nm

        energies_flat = None
        start_time = time.time()

        # ------- Priority GPU (cupy) implementation (chunked to save memory) -------
        if USE_CUPY:
            try:
                frac_atoms_cp = cp.asarray(frac_atoms)
                sigma_atoms_cp = cp.asarray(sigma_atoms)
                eps_atoms_cp = cp.asarray(eps_atoms)
                cell_cp = cp.asarray(cell)
                sigma_probe_cp = float(sigma_probe)
                eps_probe_cp = float(eps_probe)
                cutoff_cp = float(cutoff)
                min_dist_cp = float(min_dist)

                energies_flat = np.empty(frac_points.shape[0], dtype=np.float64)
                chunk = 32768
                n_points = frac_points.shape[0]

                with tqdm(total=n_points, desc="Computing energy (GPU)", unit="points") as pbar:
                    for start in range(0, n_points, chunk):
                        end = min(start + chunk, n_points)
                        pts = frac_points[start:end]
                        pts_cp = cp.asarray(pts)

                        df = frac_atoms_cp[None, :, :] - pts_cp[:, None, :]
                        df = df - cp.rint(df)
                        dr = df @ cell_cp
                        distances = cp.sqrt(cp.sum(dr ** 2, axis=2))

                        too_close = cp.any(distances < min_dist_cp, axis=1)
                        mask = (distances >= min_dist_cp) & (distances <= cutoff_cp)

                        sigma_mix = 0.5 * (sigma_atoms_cp[None, :] + sigma_probe_cp)
                        eps_mix = cp.sqrt(eps_atoms_cp[None, :] * eps_probe_cp)

                        r_eff = distances
                        sr6 = cp.zeros_like(r_eff)
                        valid = mask
                        sr6[valid] = (sigma_mix[valid] / r_eff[valid]) ** 6
                        energy_contrib = 4.0 * eps_mix[None, :] * (sr6 * (sr6 - 1.0))
                        total_energy_chunk = cp.sum(energy_contrib, axis=1)
                        total_energy_chunk[too_close] = 1000.0
                        total_energy_chunk[total_energy_chunk > 1000.0] = 1000.0

                        energies_flat[start:end] = cp.asnumpy(total_energy_chunk)
                        pbar.update(end - start)

                print("GPU computation complete (cupy)")
            except Exception as e_gpu:
                print(f"GPU acceleration failed (cupy): {e_gpu}. Falling back to CPU parallel.")
                USE_LOCAL_FALLBACK = True
            else:
                USE_LOCAL_FALLBACK = False
        else:
            USE_LOCAL_FALLBACK = True

        # ------- If GPU not used or failed, try numba parallel -------
        if USE_LOCAL_FALLBACK:
            if USE_NUMBA:
                try:
                    print("Attempting numba parallel acceleration (CPU)...")
                    energies_flat = self._compute_energies_numba(
                        frac_points.astype(np.float64),
                        frac_atoms.astype(np.float64),
                        sigma_atoms.astype(np.float64),
                        eps_atoms.astype(np.float64),
                        float(sigma_probe),
                        float(eps_probe),
                        cell.astype(np.float64),
                        float(cutoff),
                        float(min_dist)
                    )
                    print("Numba parallel computation complete")
                except Exception as e_numba:
                    print(f"Numba parallel failed: {e_numba}")
                    raise RuntimeError("Computation failed. Neither GPU nor numba acceleration available.")

        # Reshape flattened energy array to grid shape
        try:
            energies_reshaped = energies_flat.reshape((self.n1, self.n2, self.n3))
        except Exception as e:
            raise RuntimeError(f"Failed to reshape energy array: {e}")

        self.grid_energies = energies_reshaped

        # Calculate statistics (excluding placeholder 1000)
        valid_energies = self.grid_energies[self.grid_energies < 1000.0]
        if valid_energies.size > 0:
            print(f"\nEnergy statistics:")
            print(f"  Minimum: {np.min(valid_energies):.2f} kJ/mol")
            print(f"  Maximum: {np.max(valid_energies):.2f} kJ/mol")
            print(f"  Mean:    {np.mean(valid_energies):.2f} kJ/mol")
            print(f"  Std dev: {np.std(valid_energies):.2f} kJ/mol")
        else:
            print("All grid points marked as high energy (>=1000). Please verify parameters.")

        end_time = time.time()
        print(f"Computation time: {end_time - start_time:.2f} seconds")

    def save_txt(self):
        print(f"Saving TXT file: {self.out_csv.replace('.csv', '.txt')}")
        total_points = np.prod(self.n_grid)
        grid_coords = np.zeros((total_points, 3))
        grid_energies_flat = np.zeros(total_points)

        count = 0
        for ix, fx in enumerate(self.frac_x):
            for iy, fy in enumerate(self.frac_y):
                for iz, fz in enumerate(self.frac_z):
                    frac_coords = np.array([fx, fy, fz])
                    cart_coords = frac_coords @ self.cell * 10.0
                    grid_coords[count] = cart_coords
                    grid_energies_flat[count] = self.grid_energies[ix, iy, iz]
                    count += 1

        output_data = np.column_stack([grid_coords, grid_energies_flat])
        header = f"X(A) Y(A) Z(A) Energy(kJ/mol)"

        try:
            with open(self.out_csv.replace('.csv', '.txt'), "w", encoding="utf-8-sig", newline="") as f:
                np.savetxt(f, output_data, delimiter=" ", header=header,
                           comments="", fmt="%.6f")
            print(f"TXT file saved: {self.out_csv.replace('.csv', '.txt')}")
        except Exception as e:
            print(f"Failed to save TXT file: {e}")

    def save_vtk(self):
        print(f"Saving VTK file: {self.out_vtk}.vts")

        X = np.zeros((self.n1, self.n2, self.n3))
        Y = np.zeros((self.n1, self.n2, self.n3))
        Z = np.zeros((self.n1, self.n2, self.n3))

        for ix, fx in enumerate(self.frac_x):
            for iy, fy in enumerate(self.frac_y):
                for iz, fz in enumerate(self.frac_z):
                    frac_coords = np.array([fx, fy, fz])
                    cart_coords = frac_coords @ self.cell
                    X[ix, iy, iz] = cart_coords[0]
                    Y[ix, iy, iz] = cart_coords[1]
                    Z[ix, iy, iz] = cart_coords[2]

        X = np.ascontiguousarray(X)
        Y = np.ascontiguousarray(Y)
        Z = np.ascontiguousarray(Z)
        energies = np.ascontiguousarray(self.grid_energies)

        try:
            from pyevtk.hl import gridToVTK
            gridToVTK(self.out_vtk, X, Y, Z, pointData={"Energy_kJ_mol": energies})
            print(f"VTK file saved: {self.out_vtk}.vtu")
            print("  Viewable in ParaView for 3D visualization")
        except ImportError:
            try:
                from pyevtk.hl import StructuredGridToVTK
                StructuredGridToVTK(self.out_vtk, X, Y, Z, pointData={"Energy_kJ_mol": energies})
                print(f"VTK file saved: {self.out_vtk}.vts")
                print("  Viewable in ParaView for 3D visualization")
            except Exception as e2:
                print(f"Failed to save VTK file: {e2}")


class ZeolitePoreAnalyzer:
    """Zeolite channel analysis class"""

    def __init__(self, txt_file, energy_min=-20, energy_max=300,
                 cluster_eps=2.0, min_samples=10, section_interval=0.5):
        """
        Initialize channel analyzer

        Parameters
        ----------
        txt_file : str
            Potential energy data file path
        energy_min : float
            Channel region energy lower bound (kJ/mol)
        energy_max : float
            Channel region energy upper bound (kJ/mol)
        cluster_eps : float
            DBSCAN clustering neighborhood radius (A)
        min_samples : int
            DBSCAN clustering minimum sample size
        section_interval : float
            Channel cross-section spacing (A)
        """
        self.txt_file = txt_file
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.cluster_eps = cluster_eps
        self.min_samples = min_samples
        self.section_interval = section_interval

        self.data = None
        self.pore_points = None
        self.clusters = None
        self.pore_channels = []

    def load_data(self):
        """Load potential energy data file"""
        print(f"Reading file: {self.txt_file}")
        try:
            if USE_PANDAS:
                # Load using pandas
                self.data = pd.read_csv(self.txt_file, sep=r'\s+', comment='#',
                                        names=['X', 'Y', 'Z', 'Energy'])
                print(f"Successfully read {len(self.data)} data points")

                # Ensure all columns are numeric
                for col in ['X', 'Y', 'Z', 'Energy']:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

                # Remove rows containing NaN
                initial_count = len(self.data)
                self.data = self.data.dropna()
                if len(self.data) < initial_count:
                    print(f"Removed {initial_count - len(self.data)} invalid data points")

                # Display energy statistics
                print(f"\nEnergy statistics:")
                print(f"  Minimum: {self.data['Energy'].min():.2f} kJ/mol")
                print(f"  Maximum: {self.data['Energy'].max():.2f} kJ/mol")
                print(f"  Mean:    {self.data['Energy'].mean():.2f} kJ/mol")
                print(f"  Std dev: {self.data['Energy'].std():.2f} kJ/mol")
            else:
                # Load using numpy
                data_array = np.loadtxt(self.txt_file, comments='#')
                print(f"Successfully read {len(data_array)} data points")

                # Create dictionary to simulate pandas DataFrame
                self.data = {
                    'X': data_array[:, 0],
                    'Y': data_array[:, 1],
                    'Z': data_array[:, 2],
                    'Energy': data_array[:, 3]
                }

                # Display energy statistics
                energies = self.data['Energy']
                print(f"\nEnergy statistics:")
                print(f"  Minimum: {np.min(energies):.2f} kJ/mol")
                print(f"  Maximum: {np.max(energies):.2f} kJ/mol")
                print(f"  Mean:    {np.mean(energies):.2f} kJ/mol")
                print(f"  Std dev: {np.std(energies):.2f} kJ/mol")

        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

    def filter_pore_region(self):
        """Filter channel region (points with energy within specified range)"""
        print(f"\nFiltering channel region (energy range: {self.energy_min} to {self.energy_max} kJ/mol)")

        if USE_PANDAS:
            # Filter using pandas
            mask = (self.data['Energy'] >= self.energy_min) & (self.data['Energy'] <= self.energy_max)
            self.pore_points = self.data[mask].copy().reset_index(drop=True)
            total_points = len(self.data)
        else:
            # Filter using numpy
            energies = self.data['Energy']
            mask = (energies >= self.energy_min) & (energies <= self.energy_max)

            # Create filtered data dictionary
            self.pore_points = {
                'X': self.data['X'][mask],
                'Y': self.data['Y'][mask],
                'Z': self.data['Z'][mask],
                'Energy': self.data['Energy'][mask]
            }
            total_points = len(self.data['X'])

        pore_count = len(self.pore_points['X']) if not USE_PANDAS else len(self.pore_points)
        print(f"Channel region contains {pore_count} points ({pore_count / total_points * 100:.1f}%)")

        if pore_count == 0:
            raise ValueError("No points satisfying channel criteria found. Please adjust energy range.")

    def cluster_pore_points(self):
        """Cluster channel points using DBSCAN to identify independent channels"""
        print(f"\nPerforming channel clustering (eps={self.cluster_eps} A, min_samples={self.min_samples})")

        if not USE_SKLEARN:
            raise RuntimeError("sklearn not installed. Cannot perform cluster analysis.")

        # Extract coordinates
        if USE_PANDAS:
            coordinates = self.pore_points[['X', 'Y', 'Z']].values
            pore_count = len(self.pore_points)
        else:
            coordinates = np.column_stack([
                self.pore_points['X'],
                self.pore_points['Y'],
                self.pore_points['Z']
            ])
            pore_count = len(self.pore_points['X'])

        # DBSCAN clustering
        dbscan = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(coordinates)

        # Add cluster labels to data
        if USE_PANDAS:
            self.pore_points['cluster'] = cluster_labels
        else:
            self.pore_points['cluster'] = cluster_labels

        # Summarize clustering results
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise points (-1)
        n_noise = np.sum(cluster_labels == -1)

        print(f"Identified {n_clusters} independent channels")
        print(f"Noise points: {n_noise} ({n_noise / pore_count * 100:.1f}%)")

        # Display point count per channel
        for label in unique_labels:
            if label != -1:
                count = np.sum(cluster_labels == label)
                print(f"  Channel {label}: {count} points")

    def analyze_pore_channels(self):
        """Analyze geometric features of each channel"""
        print(f"\nAnalyzing channel geometric features...")

        self.pore_channels = []
        unique_labels = np.unique(self.pore_points['cluster'])

        # Set point threshold; only clusters with sufficient points undergo cross-section calculation
        min_points_threshold = 100

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            # Get points for current channel
            if USE_PANDAS:
                channel_points = self.pore_points[self.pore_points['cluster'] == label]
                coordinates = channel_points[['X', 'Y', 'Z']].values
                energies = channel_points['Energy'].values
            else:
                # Use numpy boolean indexing
                mask = self.pore_points['cluster'] == label
                coordinates = np.column_stack([
                    self.pore_points['X'][mask],
                    self.pore_points['Y'][mask],
                    self.pore_points['Z'][mask]
                ])
                energies = self.pore_points['Energy'][mask]

            # Check if sufficient points
            if len(coordinates) < min_points_threshold:
                print(f"  Channel {label}: {len(coordinates)} points (< {min_points_threshold}, skipping cross-section calculation)")
                continue

            # Calculate channel properties
            channel_info = self._calculate_channel_properties(coordinates, energies, label)
            self.pore_channels.append(channel_info)

        # Sort by average cross-section (descending)
        self.pore_channels.sort(key=lambda x: x['avg_cross_section'], reverse=True)

        print(f"\nChannel geometric feature analysis results:")
        print(
            f"{'Channel ID':<12} {'Points':<10} {'Length(A)':<12} {'AvgArea(A2)':<14} {'MedArea(A2)':<14} {'MaxArea(A2)':<14} {'MinArea(A2)':<14} {'StdArea':<12}")
        print("-" * 102)

        for i, channel in enumerate(self.pore_channels):
            print(f"{channel['id']:<12} {channel['n_points']:<10} "
                  f"{channel['length']:<12.2f} {channel['avg_cross_section']:<14.2f} "
                  f"{channel['median_cross_section']:<14.2f} {channel['max_cross_section']:<14.2f} "
                  f"{channel['min_cross_section']:<14.2f} {channel['std_cross_section']:<12.2f}")

    def _calculate_channel_properties(self, coordinates, energies, label):
        """Calculate precise geometric properties of a single channel"""
        n_points = len(coordinates)

        # Basic statistics
        center = np.mean(coordinates, axis=0)
        avg_energy = np.mean(energies)

        # Calculate channel main axis (using PCA)
        coords_centered = coordinates - center
        cov_matrix = np.cov(coords_centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

        # Main axis direction (eigenvector of maximum eigenvalue)
        main_axis = eigenvecs[:, -1]

        # Project onto main axis to calculate length
        projections = np.dot(coords_centered, main_axis)
        length = np.max(projections) - np.min(projections)

        # Accurate cross-section area: calculate real area along main axis
        cross_sections = []
        cross_section_centers = []

        if length > 0:
            # Divide channel into evenly-spaced sections along main axis
            n_sections = max(20, int(length / self.section_interval))  # At least 20 sections
            section_positions = np.linspace(np.min(projections), np.max(projections), n_sections)

            for i, pos in enumerate(section_positions[:-1]):
                # Define current section range
                pos_next = section_positions[i + 1]

                # Find points in current section range
                in_section = (projections >= pos) & (projections < pos_next)

                if np.sum(in_section) < 3:  # Too few points in section, skip
                    continue

                section_points = coordinates[in_section]

                # Calculate section center
                if len(section_points) > 0:
                    section_center_3d = np.mean(section_points, axis=0)
                else:
                    section_center_proj = (pos + pos_next) / 2
                    section_center_3d = center + section_center_proj * main_axis

                # Project points in section onto plane perpendicular to main axis
                section_coords_centered = section_points - section_center_3d
                projected_coords = section_coords_centered - np.outer(
                    np.dot(section_coords_centered, main_axis), main_axis
                )

                # Calculate cross-section area - strictly following original code logic
                if len(projected_coords) >= 3:
                    try:
                        from scipy.spatial import Delaunay
                        tri = Delaunay(projected_coords)  # Original code does not limit dimensions
                        cross_section_area = tri.volume  # In 2D, volume is the area
                    except:
                        # Fallback to bounding box approximation
                        u_axis = eigenvecs[:, -2]  # Secondary axis
                        v_axis = eigenvecs[:, -3]  # Tertiary axis
                        u_coords = np.dot(projected_coords, u_axis)
                        v_coords = np.dot(projected_coords, v_axis)

                        u_range = np.max(u_coords) - np.min(u_coords)
                        v_range = np.max(v_coords) - np.min(v_coords)
                        cross_section_area = u_range * v_range * 0.785  # Ellipse approximation
                else:
                    cross_section_area = 0.1  # Minimum area

                cross_sections.append(cross_section_area)
                cross_section_centers.append(section_center_3d.copy())

        # Calculate tortuosity
        tortuosity = 1.0
        if len(cross_section_centers) > 3:
            centerline_coords = np.array(cross_section_centers)
            vectors = np.diff(centerline_coords, axis=0)
            lengths = np.sqrt(np.sum(vectors ** 2, axis=1))
            actual_length = np.sum(lengths)
            straight_distance = np.linalg.norm(centerline_coords[-1] - centerline_coords[0])
            if straight_distance > 0:
                tortuosity = actual_length / straight_distance

        # Calculate cross-section statistics
        if cross_sections:
            cross_sections = np.array(cross_sections)
            avg_cross_section = np.mean(cross_sections)
            max_cross_section = np.max(cross_sections)
            min_cross_section = np.min(cross_sections)
            std_cross_section = np.std(cross_sections)
            median_cross_section = np.median(cross_sections)
            volume = avg_cross_section * length
        else:
            avg_cross_section = 0.0
            max_cross_section = 0.0
            min_cross_section = 0.0
            std_cross_section = 0.0
            median_cross_section = 0.0
            volume = 0.0

        # Calculate aspect ratio
        sorted_eigenvals = np.sort(eigenvals)[::-1]
        aspect_ratio = sorted_eigenvals[0] / sorted_eigenvals[1] if sorted_eigenvals[1] > 0 else 1.0

        return {
            'id': int(label),
            'n_points': n_points,
            'center': center,
            'length': length,
            'volume': volume,
            'avg_cross_section': avg_cross_section,
            'max_cross_section': max_cross_section,
            'min_cross_section': min_cross_section,
            'median_cross_section': median_cross_section,
            'std_cross_section': std_cross_section,
            'avg_energy': avg_energy,
            'aspect_ratio': aspect_ratio,
            'tortuosity': tortuosity,
            'coordinates': coordinates,
            'energies': energies,
            'main_axis': main_axis,
            'cross_sections': cross_sections if len(cross_sections) > 0 else np.array([]),
            'cross_section_centers': np.array(cross_section_centers) if len(cross_section_centers) > 0 else np.array(
                []).reshape(0, 3)
        }

    def identify_main_diffusion_channel(self):
        """Identify main diffusion channel (channel with maximum average cross-section)"""
        if not self.pore_channels:
            raise ValueError("Please run channel analysis first")

        main_channel = self.pore_channels[0]  # Already sorted by cross-section area

        print(f"\nMain diffusion channel identification results:")
        print(f"  Channel ID:        {main_channel['id']}")
        print(f"  Number of points:  {main_channel['n_points']}")
        print(f"  Channel length:    {main_channel['length']:.2f} A")
        print(f"  Avg cross-section: {main_channel['avg_cross_section']:.2f} A^2")
        print(f"  Med cross-section: {main_channel['median_cross_section']:.2f} A^2")
        print(f"  Max cross-section: {main_channel['max_cross_section']:.2f} A^2")
        print(f"  Min cross-section: {main_channel['min_cross_section']:.2f} A^2")
        print(f"  Std cross-section: {main_channel['std_cross_section']:.2f} A^2")
        print(f"  Average energy:    {main_channel['avg_energy']:.2f} kJ/mol")
        print(f"  Aspect ratio:      {main_channel['aspect_ratio']:.2f}")
        print(f"  Tortuosity:        {main_channel['tortuosity']:.4f}")

        return main_channel

    def run_analysis(self):
        """Run complete channel analysis process"""
        start_time = time.time()
        print("Starting zeolite channel analysis...")

        try:
            # Step 1: Read data
            self.load_data()

            # Step 2: Filter channel region
            self.filter_pore_region()

            # Step 3: Clustering to identify channels
            self.cluster_pore_points()

            # Step 4: Analyze channel geometry
            self.analyze_pore_channels()

            # Step 5: Identify main diffusion channel
            main_channel = self.identify_main_diffusion_channel()

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\nAnalysis complete. Main diffusion channel: Channel ID {main_channel['id']}")
            print(f"Total runtime: {elapsed_time:.2f} seconds")

            return main_channel

        except Exception as e:
            print(f"Analysis failed: {e}")
            raise


class ZeoliteFeatureExtractor:
    """Zeolite feature descriptor extractor"""

    def __init__(self, cif_file, output_dir=None, auto_remove_symmetry=True):
        """
        Initialize feature extractor

        Parameters
        ----------
        cif_file : str
            Input CIF file path
        output_dir : str
            Output directory path; uses CIF directory if None
        auto_remove_symmetry : bool
            Whether to auto-remove symmetry (default: True)
        """
        self.original_cif_file = cif_file
        self.zeolite_name = os.path.splitext(os.path.basename(cif_file))[0]

        if output_dir is None:
            self.output_dir = os.path.dirname(cif_file)
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Preprocessing: symmetry removal
        self.processed_cif_file = self._preprocess_symmetry(auto_remove_symmetry)

        # Use processed CIF file for subsequent calculations
        self.cif_file = self.processed_cif_file

        # Feature result storage
        self.features = {'zeolites': self.zeolite_name}

        # Common parameters
        self.vdw_dict = {'O': 1.319, 'Si': 1.319}
        self.ff_params = {
            "o": (0.3304, 0.442329411),
            "si": (0.2310, 0.18458107),
            "ch4": (0.3730, 1.230540467),
        }

    def _preprocess_symmetry(self, auto_remove_symmetry):
        """
        Preprocess CIF file: remove symmetry

        Parameters
        ----------
        auto_remove_symmetry : bool
            Whether to automatically remove symmetry

        Returns
        -------
        str
            Processed CIF file path
        """
        if not auto_remove_symmetry:
            print("Skipping symmetry processing")
            return self.original_cif_file

        if not USE_PYMATGEN:
            print("pymatgen not installed. Skipping symmetry processing.")
            return self.original_cif_file

        try:
            print("=" * 60)
            print("Starting preprocessing: symmetry removal")
            print("=" * 60)

            # Create symmetry processor
            symmetry_remover = SymmetryRemover()

            # Create processed CIF file path
            processed_filename = f"{self.zeolite_name}_no_symmetry.cif"
            processed_cif_path = os.path.join(self.output_dir, processed_filename)

            # Process symmetry
            result_path, success = symmetry_remover.process_cif_file(
                self.original_cif_file,
                processed_cif_path
            )

            if success:
                print("Symmetry preprocessing complete")
                print("=" * 60)
                return result_path
            else:
                print("Symmetry processing failed. Using original file.")
                print("=" * 60)
                return self.original_cif_file

        except Exception as e:
            print(f"Symmetry preprocessing error: {e}")
            print("Continuing with original file...")
            print("=" * 60)
            return self.original_cif_file

    def _get_default_config(self):
        """Get default configuration parameters"""
        return {
            'energy_grid': {
                'spacing_A': 0.2, 'cutoff_A': 14.0,
                'probe_type': 'ch4', 'min_length': 28.0
            },
            'diffusion_channel': {'energy_min': -20, 'energy_max': 300, 'cluster_eps': 0.5}
        }

    def _symbol_to_ff_key(self, symbol):
        """Map atom symbol to force field parameters"""
        clean_symbol = ''.join([c for c in symbol if c.isalpha()]).lower()
        if clean_symbol.startswith("si"):
            return "si"
        elif clean_symbol.startswith("al"):
            return "si"  # Al atoms use Si force field parameters (if Al present)
        elif clean_symbol.startswith("o"):
            return "o"
        elif clean_symbol in self.ff_params:
            return clean_symbol
        else:
            # If unrecognized, print warning and use oxygen parameters
            print(f"Warning: Unrecognized atom symbol '{symbol}'. Using oxygen force field parameters.")
            return "o"

    def generate_energy_grid(self, spacing_A=0.2, cutoff_A=14.0, probe_type="ch4", min_length=28.0):
        """
        Generate potential energy grid file (following original code logic exactly)

        Parameters
        ----------
        spacing_A : float
            Grid spacing (A)
        cutoff_A : float
            Cutoff distance (A)
        probe_type : str
            Probe molecule type
        min_length : float
            Minimum unit cell edge length threshold (A)

        Returns
        -------
        tuple
            (txt_file_path, vts_file_path)
        """
        try:
            # Create temporary potential energy grid calculator
            output_prefix = os.path.join(self.output_dir, self.zeolite_name)

            calculator = ZeolitePotentialGrid(
                cif_file=self.cif_file,
                output_prefix=output_prefix,
                spacing_A=spacing_A,
                cutoff_A=cutoff_A,
                probe_type=probe_type
            )

            print("=" * 60)
            print("Zeolite Potential Energy Grid Calculator")
            print("=" * 60)

            # Follow original code complete workflow
            calculator.expand_cif(min_length)
            calculator.load_structure()
            calculator.generate_grid()
            calculator.calculate_potential()
            calculator.save_txt()
            calculator.save_vtk()

            print("=" * 60)
            print("Potential energy grid computation complete")
            print("=" * 60)

            # Return generated file paths
            txt_file = calculator.out_csv.replace('.csv', '.txt')
            vts_file = f"{calculator.out_vtk}.vts"

            return txt_file, vts_file

        except Exception as e:
            print(f"Energy grid generation failed: {e}")
            return None, None

    def analyze_main_diffusion_channel(self, txt_file, energy_min=-20, energy_max=300, cluster_eps=0.5):
        """
        Analyze main diffusion channel features (using original ZeolitePoreAnalyzer logic)

        Parameters
        ----------
        txt_file : str
            Energy grid TXT file path
        energy_min : float
            Channel region energy lower bound
        energy_max : float
            Channel region energy upper bound

        Returns
        -------
        dict
            Main diffusion channel features
        """
        try:
            # If txt_file is None, try to construct default path
            if not txt_file:
                txt_file = os.path.join(self.output_dir, f"{self.zeolite_name}_energy_grid.txt")
                print(f"Using default energy grid file path: {txt_file}")

            print(f"Checking energy grid file: {txt_file}")
            print(f"File exists: {'Yes' if os.path.exists(txt_file) else 'No'}")

            if not os.path.exists(txt_file):
                print("Energy grid file does not exist. Skipping main diffusion channel analysis.")
                print(f"  Expected file path: {txt_file}")
                # List files in output directory for debugging
                if os.path.exists(self.output_dir):
                    files = [f for f in os.listdir(self.output_dir) if f.endswith('.txt')]
                    print(f"  TXT files in output directory: {files}")
                self.features.update({
                    'AvgA': None, 'MedA': None, 'MaxA': None,
                    'MinA': None, 'StdA': None, 'Tort': None
                })
                return None

            print(f"Energy grid file found. Starting analysis...")

            # Use original ZeolitePoreAnalyzer class for analysis
            analyzer = ZeolitePoreAnalyzer(
                txt_file=txt_file,
                energy_min=energy_min,
                energy_max=energy_max,
                cluster_eps=cluster_eps,  # Use passed parameters
                min_samples=10,
                section_interval=0.5
            )

            # Run complete analysis workflow
            main_channel = analyzer.run_analysis()

            if main_channel is None:
                print("Main diffusion channel not found")
                self.features.update({
                    'AvgA': None, 'MedA': None, 'MaxA': None,
                    'MinA': None, 'StdA': None, 'Tort': None
                })
                return None

            # Update features
            self.features.update({
                'AvgA': main_channel['avg_cross_section'],
                'MedA': main_channel['median_cross_section'],
                'MaxA': main_channel['max_cross_section'],
                'MinA': main_channel['min_cross_section'],
                'StdA': main_channel['std_cross_section'],
                'Tort': main_channel['tortuosity']
            })

            return main_channel

        except Exception as e:
            print(f"Main diffusion channel analysis failed: {e}")
            self.features.update({
                'AvgA': None, 'MedA': None, 'MaxA': None,
                'MinA': None, 'StdA': None, 'Tort': None
            })
            return None

    def extract_all_features(self, config=None):
        """
        Extract diffusion channel feature descriptors

        Parameters
        ----------
        config : dict
            Configuration parameters dictionary

        Returns
        -------
        pd.DataFrame or dict
            Feature table
        """
        print(f"\nStarting feature descriptor extraction for zeolite {self.zeolite_name}...")
        print("=" * 60)

        # Use default values if no configuration provided
        if config is None:
            config = self._get_default_config()

        # 1. Generating energy grid
        print("[1/2] Generating energy grid...")
        energy_config = config.get('energy_grid', {})
        txt_file, vts_file = self.generate_energy_grid(
            spacing_A=energy_config.get('spacing_A', 0.2),
            cutoff_A=energy_config.get('cutoff_A', 14.0),
            probe_type=energy_config.get('probe_type', 'ch4'),
            min_length=energy_config.get('min_length', 28.0)
        )

        print(f"Energy grid file: {txt_file}")

        # 2. Main diffusion channel analysis
        print("[2/2] Analyzing main diffusion channel...")
        channel_config = config.get('diffusion_channel', {})
        self.analyze_main_diffusion_channel(
            txt_file,
            energy_min=channel_config.get('energy_min', -20),
            energy_max=channel_config.get('energy_max', 300),
            cluster_eps=channel_config.get('cluster_eps', 0.5)
        )

        # Create result output
        column_order = ['zeolites', 'AvgA', 'MedA', 'MaxA', 'MinA', 'StdA', 'Tort']

        if USE_PANDAS:
            df = pd.DataFrame([self.features])

            # Ensure all columns exist
            for col in column_order:
                if col not in df.columns:
                    df[col] = None

            df = df[column_order]

            # Save as Excel
            output_excel = os.path.join(self.output_dir, f"{self.zeolite_name}_features.xlsx")
            df.to_excel(output_excel, index=False)

            print("\nFeature extraction complete!")
            print("=" * 60)
            print(f"Feature table: {output_excel}")
            if txt_file:
                print(f"Energy grid TXT: {os.path.basename(txt_file)}")
            if vts_file:
                print(f"Energy grid VTS: {os.path.basename(vts_file)}")

            return df
        else:
            # Output in CSV format
            output_csv = os.path.join(self.output_dir, f"{self.zeolite_name}_features.csv")

            with open(output_csv, 'w', encoding='utf-8') as f:
                # Write header row
                f.write(','.join(column_order) + '\n')

                # Write data row, ensuring all columns exist
                values = []
                for col in column_order:
                    value = self.features.get(col, '')
                    if value is None:
                        value = ''
                    values.append(str(value))
                f.write(','.join(values) + '\n')

            print("\nFeature extraction complete!")
            print("=" * 60)
            print(f"Feature table: {output_csv}")
            if txt_file:
                print(f"Energy grid TXT: {os.path.basename(txt_file)}")
            if vts_file:
                print(f"Energy grid VTS: {os.path.basename(vts_file)}")

            return self.features


class BatchZeoliteProcessor:
    """Batch zeolite feature extraction processor"""

    def __init__(self, input_folder, output_file, config=None):
        """
        Initialize batch processor

        Parameters
        ----------
        input_folder : str
            Input folder containing CIF files
        output_file : str
            Output Excel file path
        config : dict
            Configuration parameters
        """
        self.input_folder = input_folder
        self.output_file = output_file
        self.config = config or self._get_default_config()

        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Store all results
        self.all_results = []

    def _get_default_config(self):
        """Get default configuration"""
        return {
            'preprocessing': {'auto_remove_symmetry': True},
            'energy_grid': {
                'spacing_A': 0.2, 'cutoff_A': 14.0,
                'probe_type': 'ch4', 'min_length': 28.0
            },
            'diffusion_channel': {
                'energy_min': -20, 'energy_max': 300, 'cluster_eps': 0.5
            }
        }

    def get_cif_files(self):
        """Get all CIF files"""
        cif_files = []
        if os.path.exists(self.input_folder):
            for filename in os.listdir(self.input_folder):
                if filename.lower().endswith('.cif'):
                    cif_files.append(os.path.join(self.input_folder, filename))
        return sorted(cif_files)

    def process_single_file(self, cif_file):
        """Process a single CIF file"""
        zeolite_name = os.path.splitext(os.path.basename(cif_file))[0]

        try:
            print(f"\n{'=' * 80}")
            print(f"Processing: {zeolite_name}")
            print(f"{'=' * 80}")

            # Create temporary output directory (for intermediate files only)
            temp_output_dir = os.path.join(os.path.dirname(self.output_file), 'temp', zeolite_name)

            # Create feature extractor
            extractor = ZeoliteFeatureExtractor(
                cif_file=cif_file,
                output_dir=temp_output_dir,
                auto_remove_symmetry=self.config['preprocessing']['auto_remove_symmetry']
            )

            # Extract features (returns feature data only, no additional files generated)
            result = extractor.extract_all_features(self.config)

            # Extract feature data
            if USE_PANDAS and hasattr(result, 'iloc'):
                # pandas DataFrame
                features_dict = result.iloc[0].to_dict()
            else:
                # dict
                features_dict = result

            print(f"{zeolite_name} processing complete")

            # Clean up temporary files (keep main results)
            self._cleanup_temp_files(temp_output_dir)

            return features_dict

        except Exception as e:
            print(f"{zeolite_name} processing failed: {e}")
            # Return empty result but retain zeolite name
            empty_result = {'zeolites': zeolite_name}
            column_order = ['zeolites', 'AvgA', 'MedA', 'MaxA', 'MinA', 'StdA', 'Tort']
            for col in column_order:
                if col not in empty_result:
                    empty_result[col] = None
            return empty_result

    def _cleanup_temp_files(self, temp_dir):
        """Clean up temporary files, retaining main results only"""
        try:
            if os.path.exists(temp_dir):
                # Delete unnecessary files
                for filename in os.listdir(temp_dir):
                    if filename.endswith(('.txt', '.vts', '.vtu', '_supercell.cif', '_no_symmetry.cif')):
                        file_path = os.path.join(temp_dir, filename)
                        try:
                            os.remove(file_path)
                        except:
                            pass
        except:
            pass

    def run_batch_processing(self):
        """Run batch processing"""
        print(f"Starting batch zeolite feature extraction")
        print(f"Input folder: {self.input_folder}")
        print(f"Output file: {self.output_file}")
        print("=" * 80)

        # Get all CIF files
        cif_files = self.get_cif_files()

        if not cif_files:
            print(f"No CIF files found in {self.input_folder}")
            return

        print(f"Found {len(cif_files)} CIF files")

        # Process each file
        success_count = 0
        failed_count = 0

        for i, cif_file in enumerate(cif_files, 1):
            zeolite_name = os.path.splitext(os.path.basename(cif_file))[0]
            print(f"\n[{i}/{len(cif_files)}] Processing {zeolite_name}")

            try:
                result = self.process_single_file(cif_file)
                self.all_results.append(result)
                success_count += 1
            except Exception as e:
                print(f"Critical error: {e}")
                # Add empty result
                empty_result = {'zeolites': zeolite_name}
                column_order = ['zeolites', 'AvgA', 'MedA', 'MaxA', 'MinA', 'StdA', 'Tort']
                for col in column_order:
                    if col not in empty_result:
                        empty_result[col] = None
                self.all_results.append(empty_result)
                failed_count += 1

        # Save results
        self._save_results()

        print(f"\n{'=' * 80}")
        print(f"Batch processing complete!")
        print(f"Successfully processed: {success_count} files")
        print(f"Failed: {failed_count} files")
        print(f"Results saved to: {self.output_file}")
        print(f"{'=' * 80}")

    def _save_results(self):
        """Save batch processing results"""
        if not self.all_results:
            print("No results to save")
            return

        # Ensure column order
        column_order = ['zeolites', 'AvgA', 'MedA', 'MaxA', 'MinA', 'StdA', 'Tort']

        if USE_PANDAS:
            # Save using pandas
            df = pd.DataFrame(self.all_results)

            # Ensure all columns exist and are in correct order
            for col in column_order:
                if col not in df.columns:
                    df[col] = None

            df = df[column_order]
            df.to_excel(self.output_file, index=False)

        else:
            # Save in CSV format
            csv_file = self.output_file.replace('.xlsx', '.csv')
            with open(csv_file, 'w', encoding='utf-8') as f:
                # Write header row
                f.write(','.join(column_order) + '\n')

                # Write data rows
                for result in self.all_results:
                    values = []
                    for col in column_order:
                        value = result.get(col, '')
                        if value is None:
                            value = ''
                        values.append(str(value))
                    f.write(','.join(values) + '\n')

            print(f"Results saved to: {csv_file}")


def main():
    """Main function - Batch processing mode"""

    # Try to load configuration from config.py
    try:
        from config import FeatureConfig
        print("Loading configuration from config.py...")
        BATCH_CONFIG = FeatureConfig.get_batch_config()
    except ImportError:
        # Fallback to default configuration
        print("Using default configuration...")
        BATCH_CONFIG = {
            # Input folder - contains all CIF files
            'input_folder': os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_cifs"),

            # Output file path
            'output_file': os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "zeolites_features.xlsx"),

            # Preprocessing configuration
            'preprocessing': {
                'auto_remove_symmetry': True,
                'symprec': 0.1,
            },

            # Energy grid generation parameters
            'energy_grid': {
                'spacing_A': 0.2,
                'cutoff_A': 14.0,
                'probe_type': 'ch4',
                'min_length': 28.0,
                'min_distance_A': 0.1,
                'ff_params': {
                    "o": (0.3304, 0.442329411),  # Trappe-zeo
                    "si": (0.2310, 0.18458107),  # Trappe-zeo
                    "ch4": (0.3730, 1.230540467), # raPPE UA
                }
            },

            # Main diffusion channel analysis parameters
            'diffusion_channel': {
                'energy_min': -20,
                'energy_max': 300,
                'cluster_eps': 0.3,
                'min_samples': 10,
                'section_interval': 0.5,
                'min_points_threshold': 100,
                'min_sections': 20
            }
        }

    # Check if input folder exists
    if not os.path.exists(BATCH_CONFIG['input_folder']):
        print(f"Error: Input folder does not exist - {BATCH_CONFIG['input_folder']}")
        return

    # Create batch processor
    batch_processor = BatchZeoliteProcessor(
        input_folder=BATCH_CONFIG['input_folder'],
        output_file=BATCH_CONFIG['output_file'],
        config=BATCH_CONFIG
    )

    # Run batch processing
    try:
        batch_processor.run_batch_processing()
        print(f"\nAll files processed successfully!")
        print(f"Final results file: {BATCH_CONFIG['output_file']}")

    except Exception as e:
        print(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
