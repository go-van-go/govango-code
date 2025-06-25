import tomli
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
        self.logger = Logger(self.output_path / "log.txt")
        self.prepare_output_dirs()

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

    def _resolve_output_path(self):
        cfg = self.cfg
        name = (
            f"a{cfg.source.amplitude}_f{cfg.source.frequency}"
            f"_h{cfg.mesh.cell_size}_d{cfg.material.inclusion_density}"
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
        cfg = self.cfg

        finite_element = LagrangeElement(
            d = 3,
            n = cfg.solver.polynomial_order
        ) 

        mesh = Mesh3d(
            finite_element=finite_element,
            grid_size=cfg.mesh.cell_size,
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
        )

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

        sim = Simulator(time_stepper)
        sim.set_save_intervals(
            image=cfg.output_intervals.image,
            data=cfg.output_intervals.data,
            points=cfg.output_intervals.points,
            energy=cfg.output_intervals.energy,
        )
        sim.track_points(
            pressure=cfg.receivers.pressure,
            x=cfg.receivers.x_velocity,
            y=cfg.receivers.y_velocity,
            z=cfg.receivers.z_velocity
        )
        sim.output_path = self.output_path
        mesh.output_path = self.output_path

        if sim.visualizer:
            sim.visualizer.output_path = self.output_path

        return sim
