import os
import shutil


# split scans specified in scannetv2_{train/val/test}.txt

splits = ["train", "val", "test"]

os.makedirs("scans_transform", exist_ok=True)

for split in splits:
    print("processing", split)
    f_name = "scannetv2_{}.txt".format(split)
    f = open(f_name, "r")
    scans = f.readlines()
    os.makedirs(split, exist_ok=True)
    for scan_name in scans:
        scan = scan_name.strip()  # strip white space
        if split == "test":
            src = "scans_test/{}/{}_vh_clean_2.ply".format(scan, scan)
            dest = "{}/{}_vh_clean_2.ply".format(split, scan)
            shutil.copyfile(src, dest)
        else:
            src = "scans/{}/{}_vh_clean_2.ply".format(scan, scan)
            dest = "{}/{}_vh_clean_2.ply".format(split, scan)
            shutil.copyfile(src, dest)

            src = "scans/{}/{}_vh_clean_2.labels.ply".format(scan, scan)
            dest = "{}/{}_vh_clean_2.labels.ply".format(split, scan)
            shutil.copyfile(src, dest)

            src = "scans/{}/{}_vh_clean_2.0.010000.segs.json".format(scan, scan)
            dest = "{}/{}_vh_clean_2.0.010000.segs.json".format(split, scan)
            shutil.copyfile(src, dest)

            src = "scans/{}/{}.aggregation.json".format(scan, scan)
            dest = "{}/{}.aggregation.json".format(split, scan)
            shutil.copyfile(src, dest)

        # NOTE copy scan_transform
        src = "scans/{}/{}.txt".format(scan, scan)
        if os.path.exists(src):
            os.makedirs("scans_transform/" + scan, exist_ok=True)
            dest = "{}/{}.txt".format("scans_transform", scan)
            shutil.copyfile(src, dest)
        
print("done")
