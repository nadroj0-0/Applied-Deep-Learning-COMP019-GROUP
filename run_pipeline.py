import subprocess
import time

def format_time(seconds):
    """Converts seconds into a readable string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

scripts = ["run_data.py", "run_train_all.py", "run_evaluate_all.py"]

total_start = time.time()

for script in scripts:
    print(f"\n" + "="*50)
    print(f"🚀 Executing {script}...")
    print("="*50)
    
    script_start = time.time()
    
    # Run the script and wait for it to finish
    result = subprocess.run(["python", script], capture_output=False, text=True)
    
    script_end = time.time()
    duration = script_end - script_start
    
    if result.returncode != 0:
        print(f"\n❌ Error in {script}. Pipeline halted after {format_time(duration)}.")
        break
    else:
        print(f"\n✅ {script} completed in {format_time(duration)}")

total_end = time.time()
total_duration = total_end - total_start

print("\n" + "🏆" * 20)
print(f"FULL M5 PIPELINE COMPLETE!")
print(f"Total Execution Time: {format_time(total_duration)}")
print("Check outputs/model_comparison.csv for final results.")
print("🏆" * 20)