Test-set results only. Final model: models/merged_best.pt. Test set: datasets/fabric_merged/test (500 images).
vision_box_level/    confusion matrices, curves, summary CSVs (box-level mAP). The summary CSVs also hold train and valid rows.
vision_image_level/  PASS/WARNING/REJECT outcomes at several thresholds. Chosen settings: 0.15 / 0.5 / 0.7 (picked on validation).
advisory/            Llama replies and shape-check results, 12 classes x 3 cases (laptop).
per_source/          test results split by source (BDFD9 142 images, FabriD 358 images), if present.
These are copies; the originals stay in evaluation/. All results here are laptop results, none were measured on the KV260.
