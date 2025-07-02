import tomli
import pickle
import json
import shutil
import hashlib
import sys
from pathlib import Path
from wave_simulator.simulator import Simulator
from wave_simulator.finite_elements import LagrangeElement
from wave_simulator.mesh import Mesh3d
from wave_simulator.physics import LinearAcoustics
from wave_simulator.time_steppers import LowStorageRungeKutta
from wave_simulator.logger import Logger
from wave_simulator.input_parser import (
    InputParser,
    SourceConfig,
    MaterialConfig,
    MeshConfig,
    SolverConfig,
    ReceiversConfig,
    OutputIntervals,
)

class SimulationSetup:
    def __init__(self, config_path: Path, base_output_dir: str = "outputs"):
        self.config_path = Path(config_path)
        self.base_output_dir = Path(base_output_dir)
        self.cfg = self._load_config()
        self.output_path = self._resolve_output_path()
        self.prepare_output_dirs()
        self.logger = Logger(self.output_path / "log.txt")

    def _load_config(self):
        with open(self.config_path, "rb") as f:
            raw = tomli.load(f)
        return InputParser(
            source=SourceConfig(**raw["source"]),
            material=MaterialConfig(**raw["material"]),
            mesh=MeshConfig(**raw["mesh"]),
            solver=SolverConfig(**raw["solver"]),
            receivers=ReceiversConfig(**raw["receivers"]),
            output_intervals=OutputIntervals(**raw["output_intervals"]),
        )

    def create_mesh(self):
        cfg = self.cfg
        # create a finite element
        finite_element = LagrangeElement(
            d = 3,
            n = cfg.solver.polynomial_order
        ) 

        mesh = Mesh3d(
            finite_element=finite_element,
            grid_size=cfg.mesh.grid_size,
            box_size=cfg.mesh.box_size,
            inclusion_density=cfg.material.inclusion_density,
            inclusion_speed=cfg.material.inclusion_wave_speed,
            outer_density=cfg.material.outer_density,
            outer_speed=cfg.material.outer_wave_speed,
            source_center=cfg.source.center,
            source_radius=cfg.source.radius,
            source_amplitude=cfg.source.amplitude,
            source_frequency=cfg.source.frequency,
            inclusion_radius=cfg.mesh.inclusion_radius,
            msh_file=self.get_mesh_directory() / "mesh.msh",
        )

        # save mesh data needed for visualization
        mesh_path = self.get_mesh_directory() / "mesh.pkl"
        if not mesh_path.exists():
            self.save_mesh_visualization_data(mesh)

        return mesh

    def get_mesh_data(self, mesh):
        # Create minimal mesh data for visualization
        mesh_data = {
            'x': mesh.x,
            'y': mesh.y,
            'z': mesh.z,
            'vertex_coordinates': mesh.vertex_coordinates,
            'cell_to_vertices': mesh.cell_to_vertices,
            'nx': mesh.nx,
            'ny': mesh.ny,
            'nz': mesh.nz,
            'reference_element': mesh.reference_element,
            'initialize_gmsh': mesh.initialize_gmsh,
            'speed_per_cell': mesh.speed[0,:],  # First row only
            'density_per_cell': mesh.density[0,:]  # First row only
        }
        return mesh_data

    def save_mesh_visualization_data(self, mesh):
        mesh_data = self.get_mesh_data(mesh)
        mesh_path = self.get_mesh_directory() / "mesh.pkl"
        with open(mesh_path, 'wb') as f:
            pickle.dump(mesh_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_mesh_directory(self):
        mesh_hash = self._get_mesh_hash()
        return Path(f"inputs/meshes/{mesh_hash}") 
        
    def _get_mesh_hash(self):
        """
        Create a short hash from mesh-related parameters.
        """
        params = {
            "grid_size": self.cfg.mesh.grid_size,
            "box_size": self.cfg.mesh.box_size,
            "inclusion_radius": self.cfg.mesh.inclusion_radius,
            "source_center": self.cfg.source.center,
            "source_radius": self.cfg.source.radius,
        }
        encoded = json.dumps(params, sort_keys=True).encode()
        return hashlib.sha1(encoded).hexdigest()[:10]
       
    def _resolve_output_path(self):
        cfg = self.cfg
        name = (
            f"a{cfg.source.amplitude}_f{cfg.source.frequency}"
            f"_h{cfg.mesh.grid_size}_d{cfg.material.inclusion_density}"
            f"_c{cfg.material.inclusion_wave_speed}"
        )
        path = self.base_output_dir / name
        hash_suffix = self._hash_config()
        path = path.with_name(f"{name}_{hash_suffix}")
        # leave program if the simulation has already been run
        if path.exists():
            print(f"Simulation already exists at {path}. Exiting simulation.")
            sys.exit(0)
        return path

    def _hash_config(self):
        with open(self.config_path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:8]

    def prepare_output_dirs(self):
        (self.output_path / "data").mkdir(parents=True, exist_ok=True)
        (self.output_path / "images").mkdir(parents=True, exist_ok=True)
        shutil.copy(self.config_path, self.output_path / "parameters.toml")

    def build_simulator(self):
        # get mesh
        mesh = self.create_mesh()

        # get parameters from parameters.toml
        cfg = self.cfg

        physics = LinearAcoustics(
            mesh=mesh,
            source_center=cfg.source.center,
            source_radius=cfg.source.radius,
            source_amplitude=cfg.source.amplitude,
            source_frequency=cfg.source.frequency,
        )

        time_stepper = LowStorageRungeKutta(
            physics=physics,
            t_initial=0.0,
            t_final=cfg.solver.total_time,
        )

        sim = Simulator(time_stepper,
                        output_path=self.output_path,
                        save_image_interval = cfg.output_intervals.image,
                        save_points_interval = cfg.output_intervals.points,
                        save_data_interval = cfg.output_intervals.data,
                        save_energy_interval = cfg.output_intervals.energy,
                        pressure_reciever_locations=cfg.receivers.pressure,
                        u_velocity_reciever_locations=cfg.receivers.x_velocity,
                        v_velocity_reciever_locations=cfg.receivers.y_velocity,
                        w_velocity_reciever_locations=cfg.receivers.z_velocity,
                        mesh_directory=self.get_mesh_directory()
                        )

        return sim
