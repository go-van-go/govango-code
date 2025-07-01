from wave_simulator.setup import SimulationSetup
import sys
import cProfile

def main():
    profiler = cProfile.Profile()
    profiler.enable()
    setup = SimulationSetup(config_path=sys.argv[1])
    sim = setup.build_simulator()
    sim.run()
    profiler.disable()
    profiler.dump_stats('profile_results.prof')  # Save for analysis

if __name__ == "__main__":
    main()
