"""GSDPT head training framework for VGGT-Omega.

Ported from the Depth-Anything-3 ``gs_training`` package. Trains only the
``gs_head`` (and its adapter is frozen); the aggregator, camera, and depth heads
stay frozen and provide self-consistent, pose-free camera + depth predictions
that drive Gaussian unprojection and novel-view rendering supervision.
"""
