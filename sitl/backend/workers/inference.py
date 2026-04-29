import asyncio
import os
import sys
import re

def clean_log_line(line: str) -> str:
    """
    Strips ANSI escape sequences (colors) and Unicode box-drawing 
    characters to keep logs clean for storage and the UI.
    """
    # 1. Strip ANSI escape sequences (e.g., \x1b[32m)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    line = ansi_escape.sub('', line)
    
    # 2. Strip Unicode Box-drawing characters used by Rich/tqdm
    # Range covers: ─ │ ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼ ╰ ╯ ═ ║ etc.
    line = re.sub(r'[─│┌┐└┘├┤┬┴┼╰╯═║╔╗╚╝╠╣╦╩╬]', '', line)
    
    return line.strip()

def extract_progress(message: str):
    """
    Looks for patterns like '10/229' or 'Frame: 15' to calculate percentage.
    """
    # Pattern for tqdm style: " 15/229 [00:01<00:15...]"
    match = re.search(r'(\d+)/(\d+)', message)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        return int((current / total) * 100)
    return None

async def run_human3r_inference(job_id: str, subsample: int, seq_path: str, jobs_store: dict):
    """
    Executes demo.py, sanitizes logs, and persists them to disk.
    """
    output_dir = f"outputs/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, "inference_logs.txt")

    cmd = [
        sys.executable, "../engine.py",
        "--model_path", "../src/human3r_896L.pth",
        "--seq_path", seq_path,
        "--output_dir", output_dir,
        "--subsample", str(subsample),
        # "--save",
        "--vis_threshold", "1.5"
    ]

    # Use STDOUT to merge error messages into the same stream as logs
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )

    # Open the log file for persistent storage
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            raw_message = line.decode().strip()
            if not raw_message:
                continue

            # 1. Sanitize the message
            clean_message = clean_log_line(raw_message)
            
            if clean_message:
                # 2. Extract progress for the React ProgressBar
                progress = extract_progress(clean_message)
                if progress is not None:
                    jobs_store[job_id]["progress"] = progress

                # 3. Update the Live jobs_store (limit to last 50 lines to save RAM)
                jobs_store[job_id]["logs"].append(clean_message)
                if len(jobs_store[job_id]["logs"]) > 50:
                    jobs_store[job_id]["logs"].pop(0)

                # 4. Persist to disk and terminal
                log_file.write(clean_message + "\n")
                log_file.flush()
                print(f"[{job_id}] {clean_message}")

    await process.wait()

    if process.returncode != 0:
        jobs_store[job_id]["status"] = "failed"
        raise Exception(f"Inference process crashed with code {process.returncode}")

    return True