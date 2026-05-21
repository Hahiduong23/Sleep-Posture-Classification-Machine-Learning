import argparse, json, glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

FEATURE_COLS = [
    "ax_mean","ax_std","ax_min","ax_max",
    "ay_mean","ay_std","ay_min","ay_max",
    "az_mean","az_std","az_min","az_max",
    "mag_mean","mag_std","mag_min","mag_max",
]

def extract_window_features(df, win=48, step=24):
    arr = df[["ax","ay","az"]].to_numpy(float)
    lab = df["label"].to_numpy()
    feats, labels = [], []
    n = len(df)
    for s in range(0, n - win + 1, step):
        seg = arr[s:s+win]; seg_lab = lab[s:s+win]
        ax, ay, az = seg[:,0], seg[:,1], seg[:,2]
        mag = np.sqrt((seg**2).sum(axis=1))
        def stats(x): return [x.mean(), (x.std(ddof=1) if len(x)>1 else 0.0), x.min(), x.max()]
        feats.append(stats(ax)+stats(ay)+stats(az)+stats(mag))
        vals, cnt = np.unique(seg_lab, return_counts=True)
        labels.append(vals[np.argmax(cnt)])
    return pd.DataFrame(feats, columns=FEATURE_COLS), np.array(labels)

def main():
    desktop = Path.home() / "Desktop"
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", default=str(desktop / "mpu_data_*.csv"))
    ap.add_argument("--win", type=int, default=48)
    ap.add_argument("--step", type=int, default=24)
    ap.add_argument("--outdir", default=str(desktop / "ML_MPU6059" / "artifacts"))
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(f)[["timestamp_ms","ax","ay","az","label"]] for f in sorted(glob.glob(args.csv_glob))]
    all_df = pd.concat(frames, ignore_index=True)

    X, y = extract_window_features(all_df, win=args.win, step=args.step)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    clf = RandomForestClassifier(n_estimators=10, min_samples_leaf=2, n_jobs=-1, random_state=42)
    clf.fit(scaler.transform(Xtr), ytr)

    yhat = clf.predict(scaler.transform(Xte))
    print("\nClassification report")
    print(classification_report(yte, yhat))
    print("Confusion matrix")
    print(confusion_matrix(yte, yhat))

    dump(clf, outdir / "rf_model.joblib")
    dump(scaler, outdir / "scaler.joblib")
    (outdir / "classes.json").write_text(json.dumps(sorted(np.unique(y).tolist()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(outdir)

if __name__ == "__main__":
    main()
