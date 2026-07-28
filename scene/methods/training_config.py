"""
Training-method configuration parser.

The command line exposes several method names and component switches. This
module normalizes the method name, validates incompatible settings early, and
returns a compact config dictionary used by all training stages.
"""
from typing import Any

VALID_METHODS = {"3dgs", "absgs", "minigs", "mcmc", "improvedgs", "gns"}


def _read_bool(opt: Any, field_name: str, default_value: bool) -> bool:
    """
        Read a boolean option and reject non-boolean values after argparse parsing.
    """
    raw_value = getattr(opt, field_name, default_value)
    if isinstance(raw_value, bool):
        return raw_value
    raise ValueError("{} only accepts true/false.".format(field_name))


def build_training_method_config(opt: Any) -> dict[str, Any]:
    """
        Normalize the selected training method and resolve method component flags.
    """
    method = str(getattr(opt, "training_method", "3dgs")).strip().lower()
    if method == "default":
        method = "3dgs"
    if method not in VALID_METHODS:
        raise ValueError("Unsupported training_method: {}".format(method))
    if method == "mcmc" and int(getattr(opt, "budget", 0)) <= 0:
        raise ValueError("mcmc requires budget > 0.")

    hf_edge_weighted_loss = _read_bool(opt, "hf_edge_weighted_loss", False)
    hf_scale_aware_refinement = _read_bool(opt, "hf_scale_aware_refinement", False)
    if (hf_edge_weighted_loss or hf_scale_aware_refinement) and method != "improvedgs":
        raise ValueError("HF-GS edge/scale components currently require training_method=improvedgs.")

    if method in ("improvedgs", "gns"):
        use_las = _read_bool(opt, "use_las", True)
        use_eas = _read_bool(opt, "use_eas", True)
        use_mu = _read_bool(opt, "use_mu", True)
        use_rap = _read_bool(opt, "use_rap", True) if method == "improvedgs" else True
    else:
        use_las = False
        use_eas = False
        use_mu = False
        use_rap = False

    if hf_scale_aware_refinement and not use_las:
        raise ValueError("HF-GS scale-aware refinement requires ImprovedGS LAS (--use_las true).")

    edge_alpha_p_ref = float(getattr(opt, "hf_edge_alpha_p_ref", 0.12))
    edge_alpha_g_ref = float(getattr(opt, "hf_edge_alpha_g_ref", 0.09))
    edge_epsilon = float(getattr(opt, "hf_edge_epsilon", 1e-6))
    pidinet_tile_size = int(getattr(opt, "hf_pidinet_tile_size", 1024))
    pidinet_tile_halo = int(getattr(opt, "hf_pidinet_tile_halo", 192))
    scale_quantile = float(getattr(opt, "hf_scale_quantile", 0.75))
    scale_eta = float(getattr(opt, "hf_scale_eta", 0.2))
    scale_interval = int(getattr(opt, "hf_scale_interval", 1000))
    scale_gamma = float(getattr(opt, "hf_scale_gamma", 0.005))
    scale_min_ratio = float(getattr(opt, "hf_scale_min_ratio", 0.70))
    if edge_alpha_p_ref < 0.0 or edge_alpha_g_ref < 0.0:
        raise ValueError("HF-GS edge reference coefficients must be non-negative.")
    if edge_epsilon <= 0.0:
        raise ValueError("hf_edge_epsilon must be positive.")
    if (
        pidinet_tile_halo < 0
        or pidinet_tile_size <= 2 * pidinet_tile_halo
        or pidinet_tile_halo % 8 != 0
        or (pidinet_tile_size - 2 * pidinet_tile_halo) % 8 != 0
    ):
        raise ValueError(
            "HF-GS PiDiNet tile halo and core size must be non-negative multiples "
            "of 8, with tile_size > 2 * halo."
        )
    if not 0.0 < scale_quantile < 1.0:
        raise ValueError("hf_scale_quantile must be between 0 and 1.")
    if scale_eta < 0.0 or scale_gamma < 0.0:
        raise ValueError("HF-GS scale eta and gamma must be non-negative.")
    if scale_interval <= 0:
        raise ValueError("hf_scale_interval must be positive.")
    if not 0.0 < scale_min_ratio <= 1.0:
        raise ValueError("hf_scale_min_ratio must be in (0, 1].")

    return {
        "training_method": method,
        "use_las": use_las,
        "use_eas": use_eas,
        "use_mu": use_mu,
        "use_rap": use_rap,
        "hf_edge_weighted_loss": hf_edge_weighted_loss,
        "hf_scale_aware_refinement": hf_scale_aware_refinement,
    }
