import math
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.derived_constants import calculate_derived_constants


def test_gamete_r_conversion():
    constants = {
        "shape": "cube",
        "vol": 1.0,
        "gamete_r": 40.0,  # µm
        "vsl": 0.1,
        "sample_rate_hz": 1.0,
    }
    result = calculate_derived_constants(constants)
    assert math.isclose(result["gamete_r"], 0.04, rel_tol=1e-12)


def test_drop_r_conversion():
    constants = {"shape": "drop", "drop_r": 500}
    result = calculate_derived_constants(constants)
    assert math.isclose(result["drop_r"], 0.5, rel_tol=1e-12)


def test_number_of_sperm_calculation():
    constants = {
        "shape": "cube",
        "vol": 2.0,          # µL
        "sperm_conc": 1000.0,  # cells/mL
        "vsl": 0.1,
        "sampl_rate_hz": 1.0,
    }
    result = calculate_derived_constants(constants)
    assert result["number_of_sperm"] == 2
