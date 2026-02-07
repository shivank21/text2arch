import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Detectron2 Flowchart Element Detection")
    parser.add_argument("--frcnn_weights", type=str, required=True,
                        help="Path to Faster R-CNN model weights")
    parser.add_argument("--test_images_dir", type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output visualization images")
    parser.add_argument("--json_output_dir", type=str, required=True,
                        help="Directory to save JSON detection results")
    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.json_output_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    ))
    cfg.DATASETS.TRAIN = ("fcb_scan_train",)
    cfg.DATASETS.TEST = ("fcb_scan_test",)
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 400
    cfg.SOLVER.MAX_ITER = 1600
    cfg.SOLVER.STEPS = (500, 1500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.TEST.EVAL_PERIOD = 500

    cfg.MODEL.WEIGHTS = args.frcnn_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    predictor = DefaultPredictor(cfg)

    mdata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    mdata.thing_classes = {0: "Node", 1: "Text", 2: "Arrow", 3: "Fig Label", 4: "Node Label"}
    mdata.thing_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255)]

    inf_files = os.listdir(args.test_images_dir)

    for imageName in tqdm(inf_files):
        try:
            if not imageName.lower().endswith(".png"):
                continue
            imagePath = os.path.join(args.test_images_dir, imageName)
            im = cv2.imread(imagePath)
            outputs = predictor(im)

            model_output = outputs["instances"].to("cpu").get_fields()
            detection_results = {
                "detection": {
                    "boxes": model_output["pred_boxes"].tensor.numpy().tolist()
                        if "pred_boxes" in model_output else [],
                    "scores": model_output["scores"].numpy().tolist()
                        if "scores" in model_output else [],
                    "classes": model_output["pred_classes"].numpy().tolist()
                        if "pred_classes" in model_output else [],
                }
            }

            json_filename = f"{os.path.splitext(imageName)[0]}.json"
            json_output_path = os.path.join(args.json_output_dir, json_filename)
            with open(json_output_path, "w") as json_file:
                json.dump(detection_results, json_file, indent=4)
        except Exception as e:
            print(f"Error processing {imageName}: {e}")
            continue

    print(f"Processing complete. JSON files saved in {args.json_output_dir}")


if __name__ == "__main__":
    main()
