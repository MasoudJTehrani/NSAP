"""
SceneGraphBench — a reproducible AI2-THOR scene-graph dataset/benchmark generator.

Renders embodied (robot's-eye) views of task-relevant objects and emits, for every
frame, a PIXEL-PERFECT ground-truth scene graph straight from the simulator:
per-object category + attributes (open/closed, sliced, on/off, ...) + relations
(on/in/...). No perception model is involved in generation, so the output is a
model-agnostic benchmark: bring your own scene-graph/detector model, run it on the
frames, and score it against the ground truth.

See README.md for the step-by-step replication guide and docs/FORMAT.md for the
dataset schema.
"""

from .generator import generate, GeneratorConfig
from .scene_graph import extract_frame_graph, build_id_map
from . import vocab

__all__ = ["generate", "GeneratorConfig", "extract_frame_graph", "build_id_map", "vocab"]
__version__ = "0.1.0"
