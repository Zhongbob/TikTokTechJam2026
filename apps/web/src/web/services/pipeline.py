from __future__ import annotations

from dataclasses import asdict, dataclass

from PIL import Image
from shared_types.augmentation import AugmentationRecord
from shared_types.detection import DetectionResult
from shared_types.transforms import TransformPipeline

from web.services.factory import get_detector, get_restorer
from web.services.methods import YOLO_CLASSIFIER, TRANSFORM_REVERSAL
from web.services.transforms import apply_transform_pipeline


@dataclass
class PipelineResult:
    """Everything the results UI needs to render one end-to-end run.

    Bundles live PIL images purely for display, so this stays local to the
    web app rather than living in shared_types.
    """

    original_image: Image.Image
    transformed_image: Image.Image
    restored_image: Image.Image
    transform_pipeline: TransformPipeline
    augmentation_record: AugmentationRecord
    detection_result: DetectionResult
    restorer_is_placeholder: bool
    method: str = TRANSFORM_REVERSAL


def build_augmentation_record(source_label: str, pipeline: TransformPipeline) -> AugmentationRecord:
    """Records a (possibly multi-step) transform chain as a single manifest entry.

    A single step is still recorded as a one-element "steps" list, so the
    schema doesn't change based on pipeline length.
    """
    return AugmentationRecord(
        source=source_label,
        transform="pipeline" if pipeline else "none",
        parameters={
            "steps": [
                {"transform": spec.transform_type.value, "parameters": asdict(spec.params)}
                for spec in pipeline
            ]
        },
    )


def run_pipeline(
    image: Image.Image,
    pipeline: TransformPipeline,
    source_label: str,
    method: str = TRANSFORM_REVERSAL,
) -> PipelineResult:
    transformed_image = apply_transform_pipeline(image, pipeline)
    augmentation_record = build_augmentation_record(source_label, pipeline)

    if method == YOLO_CLASSIFIER:
        # No restoration stage — the classifier runs on the transformed image.
        detection_result = get_detector(YOLO_CLASSIFIER).predict(transformed_image)
        return PipelineResult(
            original_image=image,
            transformed_image=transformed_image,
            restored_image=transformed_image,
            transform_pipeline=pipeline,
            augmentation_record=augmentation_record,
            detection_result=detection_result,
            restorer_is_placeholder=False,
            method=YOLO_CLASSIFIER,
        )

    # transform_reversal: the ensemble scores the *transformed* image directly —
    # its fusion sub-model restores internally (use_autoencoder=True), every
    # other member sees the original. We restore here too, only to SHOW the
    # user what that restoration looks like.
    restorer = get_restorer()
    restored_image = restorer.restore(transformed_image)
    detection_result = get_detector(TRANSFORM_REVERSAL).predict(transformed_image)

    return PipelineResult(
        original_image=image,
        transformed_image=transformed_image,
        restored_image=restored_image,
        transform_pipeline=pipeline,
        augmentation_record=augmentation_record,
        detection_result=detection_result,
        restorer_is_placeholder=restorer.is_placeholder,
        method=TRANSFORM_REVERSAL,
    )
