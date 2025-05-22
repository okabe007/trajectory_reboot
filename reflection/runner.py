from __future__ import annotations
from typing import Iterator, Tuple, Dict
from core.simulation import SpermSimulation
from reflection.scenarios import cube_scenarios, drop_scenarios, spot_scenarios

def run_all(shape: str, constants: Dict) -> Iterator[Tuple[str, SpermSimulation]]:
    if shape == "cube":
        gen = cube_scenarios(constants["step_length"], constants["radius"])
    elif shape == "drop":
        gen = drop_scenarios(constants["step_length"],
                             constants["R"],
                             constants["drop_angle"])
    elif shape == "spot":
        gen = spot_scenarios(constants["step_length"],
                             constants["R_spot"],
                             constants["theta_spot"])
    else:
        raise ValueError(f"Reflection not implemented for {shape}")

    for name, params in gen:
        c = constants.copy()
        c.update(params)
        c["analysis_type"] = "reflection"
        sim = SpermSimulation(c)
        sim.simulate()
        yield name, sim
