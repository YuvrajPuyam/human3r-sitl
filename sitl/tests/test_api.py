# NOTE: This is a MANUAL end-to-end integration script, not a pytest test.
# It needs a running server (uvicorn on :8000) and a GPU, so it is intentionally
# named run_* (not test_*) so `pytest` skips it. Run it directly:
#     python tests/test_api.py
# The automated, hermetic unit tests live in test_analytics.py /
# test_action_recognition.py / test_certify.py.
import httpx
import asyncio
import os
import json

async def run_full_pipeline(video_file_path):
    async with httpx.AsyncClient(timeout=None) as client:
        # 1. TEST UPLOAD
        print(f"📤 Uploading {video_file_path}...")
        with open(video_file_path, "rb") as f:
            files = {"file": f}
            r = await client.post("http://localhost:8000/upload", files=files)
        
        job_id = r.json()["job_id"]
        print(f"✅ Upload success. Job ID: {job_id}")

        # 2. TRIGGER RUN
        print(f"🚀 Triggering pipeline (subsample=5 for speed)...")
        await client.post(f"http://localhost:8000/run/{job_id}?subsample=5")

        # 3. LISTEN TO STATUS (SSE)
        print(f"👂 Listening for progress updates...")
        async with client.stream("GET", f"http://localhost:8000/status/{job_id}") as response:
            async for line in response.aiter_lines():
                # 1. Skip empty lines or heartbeats (lines starting with ':')
                if not line or line.startswith(":"):
                    continue
                    
                if line.startswith("data:"):
                    try:
                        # 2. Extract and strip to remove any trailing \n\n
                        raw_data = line[5:].strip()
                        data = json.loads(raw_data)
                        
                        status = data.get("status")
                        stage = data.get("stage")
                        logs = data.get("logs", [])
                        
                        if logs:
                            print(f"[Stage {stage}] Status: {status} | Log: {logs[-1]}")
                        
                        if status == "completed":
                            print("\n🏆 INTEGRATION TEST PASSED!")
                            break
                    except json.JSONDecodeError:
                        print(f"⚠️  Skipping malformed line: {line}")
                        continue

if __name__ == "__main__":
    # Point this to a very short 2-3 second video for testing
    TEST_VIDEO = "../examples/example1.mp4"
    asyncio.run(run_full_pipeline(TEST_VIDEO))