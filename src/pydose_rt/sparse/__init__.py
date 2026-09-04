"""Sparse dose-calculation building blocks.

The dense layers live under :mod:`pydose_rt.layers`. This package holds the
extension points where physics corrections or learned models are injected into
the ion pipeline without densifying it — see :class:`pydose_rt.sparse.ions.IonSparseHooks`,
which is how the trained BEV correction network is attached to
:class:`pydose_rt.engine.ion_dose_engine.IonDoseEngine`.
"""
