import argparse, json, time, sys
import numpy as np
import pandas as pd
import serial
from collections import deque
from joblib import load
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout

FEATURE_COLS = [
    "ax_mean","ax_std","ax_min","ax_max",
    "ay_mean","ay_std","ay_min","ay_max",
    "az_mean","az_std","az_min","az_max",
    "mag_mean","mag_std","mag_min","mag_max",
]

def feats_from_window(win: np.ndarray) -> np.ndarray:
    ax, ay, az = win[:,0], win[:,1], win[:,2]
    mag = np.sqrt((win**2).sum(axis=1))
    def stats(x):
        if len(x) <= 1:
            return [float(np.mean(x)), 0.0, float(np.min(x)), float(np.max(x))]
        return [float(np.mean(x)), float(np.std(x, ddof=1)), float(np.min(x)), float(np.max(x))]
    return np.array(stats(ax)+stats(ay)+stats(az)+stats(mag), dtype=float)

def make_layout():
    layout = Layout()
    layout.split_column(Layout(name="top", ratio=2), Layout(name="bottom", ratio=1))
    layout["top"].split_row(Layout(name="pred", ratio=2), Layout(name="stats", ratio=1))
    return layout

def render_ui(current_pred, conf, classes, proba_vec, buf_len, win, hz, elapsed, last_change_msg):
    title = Text.assemble(("Realtime Sleep Posture", "bold"), "  ")
    pred_txt = Text.assemble(("PREDICT: ", "bold"), (f"{current_pred.upper()}", "bold green"))
    conf_txt = Text.assemble(("CONF: ", "bold"), (f"{conf*100:5.1f}%", "bold cyan"))

    # ===== Class probabilities: không cắt nhãn, bar ngắn hơn =====
    def pretty(lbl: str) -> str: return lbl.replace("_", " ")

    tbl = Table(show_header=True, header_style="bold", box=None, expand=True, padding=(0,1))
    tbl.add_column("Class", justify="left", no_wrap=True, overflow="fold", min_width=14)
    tbl.add_column("Prob", justify="right", width=6, no_wrap=True)
    tbl.add_column("Bar", justify="left", ratio=1, no_wrap=True, overflow="crop")

    for i, c in enumerate(classes):
        p = float(proba_vec[i])
        bar = "█" * int(p * 20)
        color = "green" if c == current_pred else "white"
        tbl.add_row(f"[bold]{pretty(c)}[/]", f"{p*100:5.1f}%", f"[{color}]{bar}[/]")

    pred_panel = Panel(Text.assemble(title, "\n", pred_txt, "   ", conf_txt), subtitle="Model output", border_style="bright_blue")
    proba_panel = Panel(tbl, subtitle="Class probabilities", border_style="cyan")

    t_min = int(elapsed // 60); t_sec = int(elapsed % 60)
    stats_tbl = Table(box=None, show_header=False, expand=True)
    stats_tbl.add_row("Elapsed", f"{t_min:02d}:{t_sec:02d}")
    stats_tbl.add_row("Buffer", f"{buf_len}/{win} samples")
    stats_tbl.add_row("Rate",   f"{hz:5.1f} Hz")
    stats_tbl.add_row("Event",  last_change_msg or "-")
    stats_panel = Panel(stats_tbl, subtitle="Runtime stats", border_style="magenta")

    bottom = Panel("Press [bold]Ctrl+C[/] to stop.", border_style="grey50")

    layout = make_layout()
    left = Layout(); left.split_column(pred_panel, proba_panel)
    layout["top"]["pred"].update(left)
    layout["top"]["stats"].update(stats_panel)
    layout["bottom"].update(bottom)
    return layout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--model", default="rf_model.joblib")
    ap.add_argument("--scaler", default="scaler.joblib")
    ap.add_argument("--classes", default="classes.json")
    ap.add_argument("--win", type=int, default=24)
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--smooth_k", type=int, default=1)
    args = ap.parse_args()

    clf = load(args.model)
    scaler = load(args.scaler)
    classes = np.array(json.load(open(args.classes, "r", encoding="utf-8")))

    ser = serial.Serial(args.port, args.baud, timeout=0)
    ser.readline()

    buf = deque(maxlen=args.win)
    hist = deque(maxlen=max(1, args.smooth_k))
    t0 = time.time()
    last_pred = None
    last_change_msg = ""
    samples = 0
    last_rate_update = time.time()
    samples_at_last = 0
    hz = 0.0

    with Live(
        render_ui("-", 0.0, classes, np.zeros(len(classes)), len(buf), args.win, hz, 0.0, last_change_msg),
        refresh_per_second=20,
        screen=True  # bật full-screen để lấy tối đa chiều ngang
    ) as live:
        try:
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if not line or "timestamp_ms" in line:
                    now = time.time()
                    if now - last_rate_update >= 0.5:
                        elapsed = now - t0
                        hz = (samples - samples_at_last) / (now - last_rate_update + 1e-9)
                        samples_at_last = samples
                        last_rate_update = now
                        live.update(render_ui(last_pred or "-", 0.0, classes, np.zeros(len(classes)),
                                              len(buf), args.win, hz, elapsed, last_change_msg))
                    continue

                parts = line.split(",")
                if len(parts) < 4:
                    continue
                try:
                    ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                except:
                    continue

                buf.append([ax, ay, az]); samples += 1

                if len(buf)==args.win and (samples % args.step == 0):
                    win = np.array(buf, dtype=float)
                    feat = feats_from_window(win)
                    x_df = pd.DataFrame([feat], columns=FEATURE_COLS)
                    x_s  = scaler.transform(x_df)
                    proba = clf.predict_proba(x_s)[0]
                    pred = classes[int(np.argmax(proba))]
                    hist.append(pred)
                    if len(hist) > 1:
                        vals, cnt = np.unique(hist, return_counts=True)
                        pred = vals[int(np.argmax(cnt))]
                    if pred != last_pred:
                        last_change_msg = f"→ {pred} @ {time.strftime('%H:%M:%S')}"
                    last_pred = pred
                    elapsed = time.time() - t0
                    live.update(render_ui(pred, float(np.max(proba)), classes, proba,
                                          len(buf), args.win, hz, elapsed, last_change_msg))
        except KeyboardInterrupt:
            pass
        finally:
            ser.close()
            print("\nThoát.")

if __name__ == "__main__":
    main()
