# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import numpy as np
import torch

from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor
from vit_rollout import VITAttentionRollout

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
    return predictor


def forward_image(predictor: OCRPredictor, image: np.ndarray, device: torch.device) -> np.ndarray:

    with torch.no_grad():
        processed_batches = predictor.det_predictor.pre_processor([image])
        predictor.det_predictor.model = predictor.det_predictor.model.to(device)
        out = predictor.det_predictor.model(processed_batches[0].to(device), return_model_output=True)
        seg_map = out["out_map"].to("cpu").numpy()

    # imgs = []
    # masks = []
    # preds = []
    # for batch in processed_batches:
    #     preds.extend(predictor.reco_predictor.model(batch, return_preds=True)["preds"])
    #     masks.extend(VITAttentionRollout(self.model, "min")(batch))
    #     imgs.extend(batch.detach().cpu().numpy().transpose(0, 2, 3, 1))
    # assert len(preds) == len(masks) == len(imgs)

    return seg_map
