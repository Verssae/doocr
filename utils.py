# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List
import cv2
import numpy as np
import torch
from doctr.io.elements import Word

from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor

DET_ARCHS = [
    "db_resnet50",
    "db_resnet34",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
]
RECO_ARCHS = [
    "vitstr_small",
    "vitstr_base",
]


def load_predictor(
    det_arch: str,
    reco_arch: str,
    assume_straight_pages: bool,
    straighten_pages: bool,
    bin_thresh: float,
    head_fusion: str,
    discard_ratio: float,
    device: torch.device,
) -> OCRPredictor:

    predictor = ocr_predictor(
        det_arch,
        reco_arch,
        pretrained=True,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        export_as_straight_boxes=straighten_pages,
        detect_orientation=not assume_straight_pages,
    ).to(device)
    predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
    predictor.reco_predictor.head_fusion = head_fusion
    predictor.reco_predictor.discard_ratio = discard_ratio
    return predictor


def forward_image(predictor: OCRPredictor, image: np.ndarray, device: torch.device) -> np.ndarray:

    with torch.no_grad():
        processed_batches = predictor.det_predictor.pre_processor([image])
        predictor.det_predictor.model = predictor.det_predictor.model.to(device)
        out = predictor.det_predictor.model(processed_batches[0].to(device), return_model_output=True)
        seg_map = out["out_map"].to("cpu").numpy()

    return seg_map

def word_to_image(word: Word, page: np.ndarray)-> np.ndarray:
    x_min, y_min = word["geometry"][0]
    x_max, y_max = word["geometry"][1]
    x_min, y_min, x_max, y_max = int(x_min*page.shape[1]), int(y_min*page.shape[0]), int(x_max*page.shape[1]), int(y_max*page.shape[0])
    image = page[y_min:y_max, x_min:x_max]

    return image