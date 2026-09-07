"""Gallant terrains — Isaac Lab only."""
import active_adaptation

if active_adaptation.get_backend() == "isaaclab":
    from . import hussar_terrain  # noqa: F401 — registers hussar_* terrains
else:
    raise ImportError(
        "Gallant terrains require backend='isaaclab' "
        f"(got {active_adaptation.get_backend()!r})."
    )
