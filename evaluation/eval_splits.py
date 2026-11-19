import os, csv, yaml
from ultralytics import YOLO

D = os.path.expanduser('~/Downloads/fabric_merged')
P = os.path.expanduser('~/fabric_project')

names = yaml.safe_load(open(f'{D}/data.yaml'))['names']
yaml.safe_dump({'train': f'{D}/train/images', 'val': f'{D}/valid/images',
                'test': f'{D}/test/images', 'nc': len(names), 'names': names},
               open(f'{P}/evaluation/all_splits.yaml', 'w'))

m = YOLO(f'{P}/models/merged_best.pt')
summary, per_class = [], []
for split, folder in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
    r = m.val(data=f'{P}/evaluation/all_splits.yaml', split=split, imgsz=640, batch=16,
              plots=True, project=f'{P}/evaluation/splits', name=folder,
              exist_ok=True, verbose=False)
    summary.append([folder, round(r.box.mp, 3), round(r.box.mr, 3),
                    round(r.box.map50, 3), round(r.box.map, 3)])
    for k, c in enumerate(r.box.ap_class_index):
        per_class.append([folder, r.names[int(c)], round(float(r.box.ap50[k]), 3),
                          round(float(r.box.ap[k]), 3)])

print(f"\n{'split':8}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
for s in summary:
    print(f"{s[0]:8}{s[1]:7}{s[2]:7}{s[3]:8}{s[4]:10}")

with open(f'{P}/evaluation/split_summary.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['split', 'P', 'R', 'mAP50', 'mAP50-95']); w.writerows(summary)
with open(f'{P}/evaluation/split_per_class.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['split', 'class', 'AP50', 'AP50-95']); w.writerows(per_class)
print("\nSaved CSVs in evaluation/. Plots in evaluation/splits/{train,valid,test}/")
