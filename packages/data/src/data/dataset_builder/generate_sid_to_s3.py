"""Generate a resumable SID autoencoder dataset directly into Amazon S3.

Existing dataset builders are not modified.

Run from packages/data/src:

python -m data.generate_sid_to_s3 \
    --bucket YOUR_BUCKET \
    --prefix autoencoder/run-001 \
    --images-per-label 100 \
    --num-augmentations 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from io import BytesIO
import time

import boto3
import numpy as np
from botocore.config import Config
from botocore.exceptions import ClientError
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from data.augmentation.image_augmenter import ImageAugmenter


LABEL_NAMES = {
    0: "real",
    1: "synthetic",
    2: "tampered",
}

SID_DATASET = "saberzl/SID_Set"

# Pinning the dataset version helps ensure that restarting selects
# the same source images.
SID_REVISION = "dc03ead57929879319ce30a82bfcfb8d317b10bd"


class S3Writer:
    """Synchronously upload generated files directly from memory."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")

        self.client = boto3.client(
            "s3",
            config=Config(
                retries={
                    "max_attempts": 10,
                    "mode": "standard",
                }
            ),
        )

    def key(self, relative_path: str) -> str:
        relative_path = relative_path.lstrip("/")

        if self.prefix:
            return f"{self.prefix}/{relative_path}"

        return relative_path

    def uri(self, relative_path: str) -> str:
        return f"s3://{self.bucket}/{self.key(relative_path)}"

    def exists(self, relative_path: str) -> bool:
        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=self.key(relative_path),
            )
            return True

        except ClientError as error:
            status = error.response.get(
                "ResponseMetadata", {}
            ).get("HTTPStatusCode")

            code = error.response.get(
                "Error", {}
            ).get("Code")

            if status == 404 or code in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                return False

            raise

    def upload_png(
        self,
        image_array: np.ndarray,
        relative_path: str,
    ) -> str:
        """Encode and upload a PNG without saving it locally."""

        with BytesIO() as buffer:
            Image.fromarray(image_array).save(
                buffer,
                format="PNG",
            )

            self.client.put_object(
                Bucket=self.bucket,
                Key=self.key(relative_path),
                Body=buffer.getvalue(),
                ContentType="image/png",
            )

        return self.uri(relative_path)

    def upload_json(
        self,
        value: dict,
        relative_path: str,
    ) -> str:
        body = json.dumps(
            value,
            indent=2,
        ).encode("utf-8")

        self.client.put_object(
            Bucket=self.bucket,
            Key=self.key(relative_path),
            Body=body,
            ContentType="application/json",
        )

        return self.uri(relative_path)

    def download_json(
        self,
        relative_path: str,
    ) -> dict:
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=self.key(relative_path),
        )

        return json.loads(
            response["Body"].read().decode("utf-8")
        )


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x")
        return int(width), int(height)

    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use WIDTHxHEIGHT, for example 256x256"
        ) from error


def deterministic_seed(
    image_id: str,
    base_seed: int,
) -> int:
    """Give each SID image the same seed after every restart."""

    value = f"{base_seed}:{image_id}".encode("utf-8")
    digest = hashlib.sha256(value).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


def safe_sample_id(
    label: int,
    image_id: str,
) -> str:
    """Create a unique filename without label subdirectories."""

    readable = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        image_id,
    )[:40]

    # Include the label when hashing to prevent collisions,
    # but do not create a label directory.
    unique_value = f"{label}:{image_id}"

    digest = hashlib.sha256(
        unique_value.encode("utf-8")
    ).hexdigest()[:12]

    return f"{readable}-l{label}-{digest}"


def create_crop_target(
    augmenter: ImageAugmenter,
    clean_image: Image.Image,
    clean_array: np.ndarray,
    augmentation_record,
) -> np.ndarray:
    """Apply only the augmentation's crop to the clean target."""

    target_array = clean_array

    for step in augmentation_record.parameters.get(
        "steps", []
    ):
        if step["transform"] != "center_crop":
            continue

        crop_ratio = float(
            step["parameters"]["crop_ratio"]
        )

        crop_result = augmenter.center_crop(
            clean_image,
            crop_ratio=crop_ratio,
        )

        # Supports both:
        # return cropped_image
        # return cropped_image, parameters
        if isinstance(crop_result, tuple):
            cropped_image = crop_result[0]
        else:
            cropped_image = crop_result

        target_array = np.asarray(
            cropped_image,
            dtype=np.uint8,
        )

        break

    return target_array


def process_one_image(
    image: Image.Image,
    metadata: dict,
    writer: S3Writer,
    output_size: tuple[int, int],
    num_augmentations: int,
    base_seed: int,
) -> str:
    """Transform and upload one source image.

    The record JSON is written last and acts as the completion marker.
    """

    image_id = metadata["img_id"]
    label = metadata["sid_label"]

    sample_id = safe_sample_id(
        label=label,
        image_id=image_id,
    )

    record_path = f"records/{sample_id}.json"

    # Already completed during an earlier run.
    if writer.exists(record_path):
        return "skipped"

    image_seed = deterministic_seed(
        image_id=image_id,
        base_seed=base_seed,
    )

    augmenter = ImageAugmenter(
        output_size=output_size,
        seed=image_seed,
    )

    clean_image = image.convert("RGB").resize(
        output_size,
        Image.Resampling.LANCZOS,
    )

    clean_array = np.asarray(
        clean_image,
        dtype=np.uint8,
    )

    augmented_array, augmentation_record = (
        augmenter.transform_one(
            image,
            num_augmentations=num_augmentations,
        )
    )

    augmented_target = create_crop_target(
        augmenter=augmenter,
        clean_image=clean_image,
        clean_array=clean_array,
        augmentation_record=augmentation_record,
    )

    upload_started = time.perf_counter()

    clean_input_path = (
        f"inputs/{sample_id}_clean.png"
    )
    clean_target_path = (
        f"targets/{sample_id}_clean.png"
    )
    augmented_input_path = (
        f"inputs/{sample_id}_augmented.png"
    )
    augmented_target_path = (
        f"targets/{sample_id}_augmented.png"
    )

    # These calls are synchronous. Each upload finishes before continuing.
    clean_input_uri = writer.upload_png(
        clean_array,
        clean_input_path,
    )

    clean_target_uri = writer.upload_png(
        clean_array,
        clean_target_path,
    )

    augmented_input_uri = writer.upload_png(
        augmented_array,
        augmented_input_path,
    )

    augmented_target_uri = writer.upload_png(
        augmented_target,
        augmented_target_path,
    )

    record = {
        "sample_id": sample_id,
        "source_metadata": metadata,
        "augmentation_seed": image_seed,
        "clean_pair": {
            "input_path": clean_input_uri,
            "target_path": clean_target_uri,
            "transform": "identity",
        },
        "augmented_pair": {
            "input_path": augmented_input_uri,
            "target_path": augmented_target_uri,
            "transform": augmentation_record.transform,
            "parameters": augmentation_record.parameters,
        },
    }

    # Commit marker: uploaded only after all four PNGs succeed.
    writer.upload_json(
        record,
        record_path,
    )

    upload_seconds = (
        time.perf_counter() - upload_started
    )

    return "uploaded"


def validate_run_configuration(
    writer: S3Writer,
    run_config: dict,
) -> None:
    """Prevent incompatible runs from sharing an S3 prefix."""

    config_path = "_run_config.json"

    if writer.exists(config_path):
        existing = writer.download_json(config_path)

        if existing != run_config:
            raise RuntimeError(
                "This S3 prefix contains a different run configuration. "
                "Use a new --prefix or restore the original settings."
            )

        print("Compatible existing run found. Resuming.")
        return

    writer.upload_json(
        run_config,
        config_path,
    )

    print("New S3 run created.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream SID images, augment them and upload "
            "directly to S3."
        )
    )

    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket name without s3://",
    )

    parser.add_argument(
        "--prefix",
        required=True,
        help="Unique S3 prefix for this dataset run",
    )

    parser.add_argument(
        "--images-per-label",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-augmentations",
        type=int,
        choices=range(1, 7),
        default=6,
    )

    parser.add_argument(
        "--output-size",
        type=parse_size,
        default=(256, 256),
        metavar="WIDTHxHEIGHT",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of raw Hugging Face entries to skip",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.images_per_label < 1:
        raise ValueError(
            "images-per-label must be at least 1"
        )

    writer = S3Writer(
        bucket=args.bucket,
        prefix=args.prefix,
    )

    

    run_config = {
        "dataset": SID_DATASET,
        "revision": SID_REVISION,
        "images_per_label": args.images_per_label,
        "num_augmentations": args.num_augmentations,
        "output_size": list(args.output_size),
        "seed": args.seed,
    }

    validate_run_configuration(
        writer,
        run_config,
    )

    run_started = time.perf_counter()
    previous_image_finished = run_started

    selected_counts = {
        label: 0 for label in LABEL_NAMES
    }

    uploaded_counts = {
        label: 0 for label in LABEL_NAMES
    }

    skipped_counts = {
        label: 0 for label in LABEL_NAMES
    }

    total_required = (
        args.images_per_label * len(LABEL_NAMES)
    )

    progress_bar = tqdm(
        total=total_required,
        unit="image",
        desc="SID → S3",
    )

    sid_stream = load_dataset(
        SID_DATASET,
        split="train",
        streaming=True,
        revision=SID_REVISION,
    ).shuffle(
        seed=args.seed,
        buffer_size=args.shuffle_buffer,
    )

    if args.skip > 0:
        print(
            f"Skipping the first {args.skip} "
            f"raw Hugging Face entries."
        )
    sid_stream = sid_stream.skip(args.skip)

    for example in sid_stream:
        label = int(example["label"])

        if label not in LABEL_NAMES:
            continue

        if (
            selected_counts[label]
            >= args.images_per_label
        ):
            continue

        selected_counts[label] += 1

        metadata = {
            "img_id": str(example["img_id"]),
            "sid_label": label,
            "label_name": LABEL_NAMES[label],
            "binary_aigc_label": int(label != 0),
        }

        hf_seconds = (
            time.perf_counter() - previous_image_finished
        )

        result = process_one_image(
            image=example["image"].convert("RGB"),
            metadata=metadata,
            writer=writer,
            output_size=args.output_size,
            num_augmentations=args.num_augmentations,
            base_seed=args.seed,
        )

        if result == "uploaded":
            uploaded_counts[label] += 1
        else:
            skipped_counts[label] += 1

        selected_total = sum(
            selected_counts.values()
        )

        writer.upload_json(
            {
                "selected_counts": selected_counts,
                "uploaded_this_session": uploaded_counts,
                "skipped_this_session": skipped_counts,
                "selected_total": selected_total,
                "required_total": total_required,
            },
            "progress.json",
        )

        print(
            f"Selected {selected_total}/{total_required} | "
            f"Uploaded: {uploaded_counts} | "
            f"Skipped: {skipped_counts}"
        )

        if all(
            selected_counts[label]
            >= args.images_per_label
            for label in LABEL_NAMES
        ):
            break

        progress_bar.close()

        elapsed = time.perf_counter() - run_started

        print(
            f"Dataset generation complete in "
            f"{elapsed / 60:.2f} minutes."
        )


if __name__ == "__main__":
    main()
