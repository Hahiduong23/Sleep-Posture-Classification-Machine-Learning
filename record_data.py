import argparse, serial, csv, time
from datetime import datetime
from pathlib import Path

def human_mmss(sec: float) -> str:
    sec = int(sec); m, s = divmod(sec, 60); return f"{m:02d}:{s:02d}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--label", default=None)
    p.add_argument("--outfile", default=None)
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    desktop = Path.home() / "Desktop"
    default_name = f"mpu_data_{args.label}_{ts}.csv" if args.label else f"mpu_data_{ts}.csv"
    outpath = Path(args.outfile) if args.outfile else (desktop / default_name)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    ser = serial.Serial(args.port, args.baud, timeout=1)
    _ = ser.readline().decode(errors="ignore")

    base_cols = ["timestamp_ms", "ax", "ay", "az"]
    with outpath.open("w", newline="") as f:
        writer = csv.writer(f)
        cols = base_cols + (["label"] if args.label else [])
        writer.writerow(cols)
        print(f"Ghi vào: {outpath}  (nhấn Ctrl+C để dừng)")
        start = time.time(); last_ui = 0.0; count = 0
        try:
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if not line or "timestamp_ms" in line:
                    now = time.time()
                    if now - last_ui >= 0.5:
                        elapsed = now - start
                        rate = (count / elapsed) if elapsed > 0 else 0.0
                        print(f"\r⏱ {human_mmss(elapsed)}  |  samples: {count}  |  ~{rate:5.1f} Hz", end="", flush=True)
                        last_ui = now
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4: continue
                row = parts[:4]
                if args.label: row.append(args.label)
                writer.writerow(row); count += 1
                now = time.time()
                if now - last_ui >= 0.5:
                    elapsed = now - start
                    rate = (count / elapsed) if elapsed > 0 else 0.0
                    print(f"\r⏱ {human_mmss(elapsed)}  |  samples: {count}  |  ~{rate:5.1f} Hz", end="", flush=True)
                    last_ui = now
        except KeyboardInterrupt:
            elapsed = time.time() - start
            rate = (count / elapsed) if elapsed > 0 else 0.0
            print(f"\nKết thúc ghi. Tổng thời gian: {human_mmss(elapsed)} | tổng mẫu: {count} | ~{rate:.1f} Hz")
        finally:
            ser.close()

if __name__ == "__main__":
    main()
