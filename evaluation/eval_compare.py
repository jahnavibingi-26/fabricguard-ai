"""Compare merged_best.pt (baseline) with merged_aug_best.pt (stronger augmentation) on valid and test.

Run inside ~/yolo-venv:
    python ~/fabric_project/evaluation/eval_compare.py

Needs ~/fabric_project/evaluation/all_splits.yaml (made by eval_splits.py) and both .pt files in models/.
Outputs a table in the terminal and ~/fabric_project/evaluation/compare_merged_vs_aug.csv
"""
import csv
import os

from ultralytics import YOLO

P = os.path.expanduser('~/fabric_project')
YAML = f'{P}/evaluation/all_splits.yaml'
MODELS = {'merged': f'{P}/models/merged_best.pt', 'merged_aug': f'{P}/models/merged_aug_best.pt'}
SPLITS = [('valid', 'val'), ('test', 'test')]
WEAK = ['Hole', 'Needle Mark', 'Missing Yarn', 'Stain']

for name, path in MODELS.items():
    if not os.path.exists(path):
        raise SystemExit(f'Missing {path}')

summary, per_class = {}, {}
for mname, path in MODELS.items():
    m = YOLO(path)
    for label, split in SPLITS:
        r = m.val(data=YAML, split=split, imgsz=640, batch=16, plots=False, verbose=False,
                  project=f'{P}/evaluation/compare', name=f'{mname}_{label}', exist_ok=True)
        summary[(mname, label)] = (float(r.box.mp), float(r.box.mr), float(r.box.map50), float(r.box.map))
        per_class[(mname, label)] = {r.names[int(c)]: float(r.box.ap50[k]) for k, c in enumerate(r.box.ap_class_index)}

print(f"\n{'model':12}{'split':7}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
for (mname, label), v in summary.items():
    print(f"{mname:12}{label:7}{v[0]:7.3f}{v[1]:7.3f}{v[2]:8.3f}{v[3]:10.3f}")

print(f"\nPer-class AP50 on test (all classes)   {'merged':>8}{'aug':>8}{'diff':>8}")
rows = []
for cls in per_class[('merged', 'test')]:
    a, b = per_class[('merged', 'test')][cls], per_class[('merged_aug', 'test')].get(cls, 0.0)
    flag = '  <- weak class' if cls in WEAK else ''
    print(f"  {cls:30}{a:8.3f}{b:8.3f}{b - a:+8.3f}{flag}")
    rows.append([cls, round(a, 3), round(b, 3), round(b - a, 3)])

dv = summary[('merged_aug', 'valid')][2] - summary[('merged', 'valid')][2]
dt = summary[('merged_aug', 'test')][2] - summary[('merged', 'test')][2]
weak_gain = sum(per_class[('merged_aug', 'test')][c] - per_class[('merged', 'test')][c] for c in WEAK) / len(WEAK)
print(f"\nvalid mAP50 change: {dv:+.3f}   test mAP50 change: {dt:+.3f}   mean weak-class AP50 change (test): {weak_gain:+.3f}")
print("Rule: switch only if valid mAP50 improves by >= 0.03 AND test improves AND the weak classes improve.")
print("SWITCH to merged_aug" if (dv >= 0.03 and dt > 0 and weak_gain > 0) else "KEEP merged (baseline)")

with open(f'{P}/evaluation/compare_merged_vs_aug.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['class', 'merged_test_AP50', 'aug_test_AP50', 'diff'])
    w.writerows(rows)
