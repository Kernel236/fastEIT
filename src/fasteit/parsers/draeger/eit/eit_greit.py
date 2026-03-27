"""GREIT image reconstruction for Dräger `.eit` transimpedance data.

Wraps pyEIT's GREIT implementation to convert ``RawImpedanceData.measurements``
(shape ``(N_frames, 208)``) into reconstructed 32×32 pixel images
(shape ``(N_frames, 32, 32)``).

Algorithm reference:
    Adler A, Arnold JH, Bayford R, et al.
    "GREIT: a unified approach to 2D linear EIT reconstruction of lung images."
    *Physiol. Meas.* 30 (2009) S35–S55.
    DOI: 10.1088/0967-3334/30/6/S03

Default parameters (p=0.2, lamb=1e-2, n=32) are those recommended in the
GREIT paper for lung monitoring with 16-electrode adjacent-drive protocols.
"""

from __future__ import annotations

import numpy as np

try:
    import pyeit.eit.greit as greit_mod
    import pyeit.eit.protocol as proto_mod
    import pyeit.mesh

    _PYEIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYEIT_AVAILABLE = False


def _check_pyeit() -> None:
    if not _PYEIT_AVAILABLE:
        raise ImportError(
            "pyEIT is required for image reconstruction. "
            "Install it with: pip install pyeit"
        )


def build_greit(
    n_el: int = 16,
    h0: float = 0.1,
    p: float = 0.2,
    lamb: float = 1e-2,
    n: int = 32,
) -> tuple:
    """Build and return a configured GREIT solver.

    Creates a circular unit-disk mesh, an adjacent-drive protocol (dist_exc=1,
    step_meas=1, parser='std'), and a GREIT solver ready for ``solve()``.

    Args:
        n_el:  Number of electrodes. Default 16 (Dräger PulmoVista 500).
        h0:    Mesh density parameter passed to ``pyeit.mesh.create``.
               Default 0.1.
        p:     GREIT noise figure (0–1).
               Default 0.2 (Adler 2009 recommendation).
        lamb:  Tikhonov regularisation parameter. Default 1e-2.
        n:     Output image size (``n × n`` pixels). Default 32.

    Returns:
        Tuple ``(greit_solver, protocol)`` ready for reconstruction.

    References:
        Adler et al., *Physiol. Meas.* 30 (2009) S35–S55.
        DOI: 10.1088/0967-3334/30/6/S03
    """
    _check_pyeit()
    mesh_obj = pyeit.mesh.create(n_el=n_el, h0=h0)
    protocol = proto_mod.create(n_el=n_el, dist_exc=1, step_meas=1, parser_meas="std")
    solver = greit_mod.GREIT(mesh_obj, protocol)
    solver.setup(p=p, lamb=lamb, n=n)
    return solver, protocol


def reconstruct_greit(
    measurements: np.ndarray,
    ref_frame: np.ndarray | int | tuple[int, int] | None = None,
    n_el: int = 16,
    p: float = 0.2,
    lamb: float = 1e-2,
    n: int = 32,
) -> np.ndarray:
    """Reconstruct EIT images from transimpedance measurements using GREIT.

    Converts ``RawImpedanceData.measurements`` (shape ``(N_frames, 208)``)
    to reconstructed 32×32 pixel images via the GREIT algorithm.

    **Baseline selection** — GREIT produces *differential* images relative to a
    reference frame. The default (``ref_frame=None``, mean of all frames) is the
    right choice for single-file analysis (tidal variation, regional distribution).

    **EELI and cross-recording comparison** — End-Expiratory Lung Impedance (EELI)
    is the only clinically meaningful *absolute* value in EIT: it reflects the
    functional residual capacity (FRC). To compare EELI across recordings (e.g.
    two NIV interfaces in the same patient), all files must share the same external
    reference so that ``global_signal[end_exp]`` is on the same absolute scale::

        # Record a stable baseline at the start of the session (before any
        # intervention). Use its end-expiratory mean as the reference for all
        # subsequent recordings in that session.
        ref = data_baseline.measurements[:50].mean(axis=0)  # shape (208,)

        images_niv1 = reconstruct_greit(data_niv1.measurements, ref_frame=ref)
        images_niv2 = reconstruct_greit(data_niv2.measurements, ref_frame=ref)
        # EELI_niv2 > EELI_niv1 → interface 2 maintains better FRC

    **Limitation**: if the device recalibrates between recordings, the absolute
    impedance baseline shifts and introduces an unrecoverable offset. Comparisons
    remain valid as long as the clinical effect of interest is larger than the
    calibration drift.

    Args:
        measurements: Calibrated transimpedance array, shape ``(N_frames, 208)``.
                      Typically ``RawImpedanceData.measurements``.
        ref_frame:    Reference (baseline) for differential reconstruction:
                      - ``None``             → mean of all frames in this recording.
                      - ``int``              → single frame index within this recording.
                      - ``(start, end)``     → mean of ``measurements[start:end]``.
                      - ``np.ndarray (208,)``→ external reference (from another
                        recording, or a manually defined baseline). Use this to
                        compare recordings taken at different times or after
                        device recalibration.
        n_el:         Number of electrodes. Default 16.
        p:            GREIT noise figure. Default 0.2.
        lamb:         Regularisation parameter. Default 1e-2.
        n:            Output image side length in pixels. Default 32.

    Returns:
        Reconstructed images, shape ``(N_frames, n, n)``. Values are
        differential conductivity change (Δσ), NaN outside the electrode circle.

    Raises:
        ImportError: If pyEIT is not installed.
        ValueError:  If ``measurements`` does not have shape ``(N, 208)``.

    References:
        Adler et al., *Physiol. Meas.* 30 (2009) S35–S55.
        DOI: 10.1088/0967-3334/30/6/S03
    """
    _check_pyeit()

    if measurements.ndim != 2 or measurements.shape[1] != 208:
        raise ValueError(
            f"measurements must have shape (N_frames, 208), got {measurements.shape}"
        )

    # Resolve reference frame (v0)
    if ref_frame is None:
        v0 = measurements.mean(axis=0)  # mean over all frames
    elif isinstance(ref_frame, int):
        v0 = measurements[ref_frame]
    elif isinstance(ref_frame, tuple):
        start, end = ref_frame
        v0 = measurements[start:end].mean(axis=0)  # mean over frame range
    else:
        v0 = np.asarray(ref_frame)
        if v0.shape != (208,):
            raise ValueError(f"ref_frame array must have shape (208,), got {v0.shape}")

    solver, _ = build_greit(n_el=n_el, p=p, lamb=lamb, n=n)

    n_frames, n_meas = measurements.shape  # (N_frames, 208)
    images = np.full((n_frames, n, n), np.nan)  # output: (N_frames, n, n)

    for i, v1 in enumerate(measurements):
        # GREIT computes Δσ (conductivity change): decreases when air enters the lungs.
        # Clinical EIT convention is the opposite (impedance increases with air).
        # Negate so images match the .bin Dräger convention — peaks up during inspiration.
        # Rotate 90° CCW (k=1) to match anatomical orientation: heart left-anterior,
        # right lung on the right (same as a CT cross-section viewed from the feet).
        ds = solver.solve(v1, v0)
        images[i] = np.rot90(-ds.reshape(n, n), k=1)

    return images
