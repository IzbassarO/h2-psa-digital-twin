import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_check_params_passes():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_params.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_virial_mixture_reduces_to_pure():
    sys.path.insert(0, str(ROOT / "src"))
    from h2psa.isotherms import Virial

    v = Virial.from_yaml(ROOT / "data" / "params" / "ac5ks_lopes2010.yaml")
    q_pure = v.loading_pure("CO2", 3.0, 303)
    q_mix = v.loading_mix({"CO2": 1.0}, 3.0, 303)["CO2"]
    assert abs(q_mix / q_pure - 1) < 1e-6


def test_msl_mixture_reduces_to_pure():
    sys.path.insert(0, str(ROOT / "src"))
    from h2psa.isotherms import MultisiteLangmuir

    m = MultisiteLangmuir.from_yaml(ROOT / "data" / "params" / "hy2seps_zeo_lopes2010.yaml")
    assert abs(m.loading_mix({"N2": 1.0}, 5.0, 303)["N2"] / m.loading_pure("N2", 5.0, 303) - 1) < 1e-6
