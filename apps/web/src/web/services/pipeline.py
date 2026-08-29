from __future__ import annotations

from dataclasses import asdict, dataclass

from PIL import Image
from shared_types.augmentation import AugmentationRecord
from shared_types.detection import DetectionResult
from shared_types.transforms import TransformPipeline

from web.services.factory import get_detector, get_restorer
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


def run_pipeline(image: Image.Image, pipeline: TransformPipeline, source_label: str) -> PipelineResult:
    transformed_image = apply_transform_pipeline(image, pipeline)

    restorer = get_restorer()
    restored_image = restorer.restore(transformed_image)

    detector = get_detector()
    detection_result = detector.predict(restored_image)

    augmentation_record = build_augmentation_record(source_label, pipeline)

    return PipelineResult(
        original_image=image,
        transformed_image=transformed_image,
        restored_image=restored_image,
        transform_pipeline=pipeline,
        augmentation_record=augmentation_record,
        detection_result=detection_result,
        restorer_is_placeholder=restorer.is_placeholder,
    )
