"""Pixel-level anomaly detection on the Spacepresso dataset.

    from spacepresso.detectors import get_detector
    from spacepresso.runner import RuntimeConfig, run_experiment

    PatchCore, PatchCoreConfig = get_detector("patchcore")
    run_experiment(PatchCore, PatchCoreConfig(backbone="dinov2_vitb14_reg"),
                   RuntimeConfig())

Layer order, strictly inward — ``core`` imports nothing from the project,
``detectors`` never import each other, and ``tests/unit/test_architecture.py``
enforces both::

    core  <-  data, backbones  <-  detectors, postprocess  <-  stacking, runner
"""

__version__ = "1.0.0"
