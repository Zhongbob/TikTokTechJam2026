"""Back-compat shim: the meta-classifier now lives in `detector_common.meta`
(shared by fusion and ensemble). Import from there instead."""

from detector_common.meta import (
    FEATURE_SPECS,
    MetaClassifier,
    build_feature_matrix,
    default_estimator,
)

__all__ = ["FEATURE_SPECS", "MetaClassifier", "build_feature_matrix", "default_estimator"]
