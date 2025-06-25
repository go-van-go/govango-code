from wave_simulator.setup import SimulationSetup
import sys

def main():
    setup = SimulationSetup(config_path=sys.argv[1])
    sim = setup.build_simulator()
    sim.run()

if __name__ == "__main__":
    main()
