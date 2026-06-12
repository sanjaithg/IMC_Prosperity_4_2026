import glob
import subprocess
import os
import concurrent.futures

submissions = sorted(glob.glob("submissions/submission_*.py"))
results = {}

def run_backtest(sub):
    print(f"Starting {sub}...", flush=True)
    cmd = ["../venv/bin/python", "backtester.py", "--submission", sub, "--days", "2", "3", "4", "--quiet"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = proc.stdout.split('\n')
        final_pnl = "N/A"
        for line in lines:
            if "Total across 3 day(s):" in line:
                final_pnl = line.split(":")[-1].strip()
        print(f"Finished {sub}: {final_pnl}", flush=True)
        return sub, final_pnl
    except Exception as e:
        print(f"Failed {sub}: {e}", flush=True)
        return sub, f"ERROR: {e}"

# Run up to 4 concurrent backtests
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(run_backtest, sub): sub for sub in submissions}
    for future in concurrent.futures.as_completed(futures):
        sub, pnl = future.result()
        results[sub] = pnl

# Write markdown report
with open("backtest_report.md", "w") as f:
    f.write("# Backtest Results (Days 2, 3, 4)\n\n")
    f.write("| Submission | 3-Day Total PnL |\n")
    f.write("| :--- | :--- |\n")
    for sub in submissions:
        pnl = results.get(sub, "N/A")
        f.write(f"| `{os.path.basename(sub)}` | **{pnl}** |\n")

print("\nAll backtests completed. Results saved to backtest_report.md")
