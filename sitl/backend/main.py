from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
import uuid
import os
import shutil
import json

from .pipeline import run_pipeline

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs("outputs",   exist_ok=True)

app = FastAPI(title="SITL Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store — fine for local single-user use
jobs: dict = {}

# Serve output files (PLY, JSON) directly to Three.js
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Step 1: receive video, persist to disk, return job_id."""
    job_id    = str(uuid.uuid4())[:8]
    extension = os.path.splitext(file.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{extension}")

    with open(video_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    jobs[job_id] = {
        "status":     "uploaded",
        "video_path": video_path,
        "filename":   file.filename,
        "stage":      0,
        "progress":   0,
        "logs":       [f"Uploaded: {file.filename}"],
    }
    return {"job_id": job_id}


@app.post("/run/{job_id}")
async def start_pipeline(
    job_id: str,
    background_tasks: BackgroundTasks,
    subsample: int = 1,
):
    """Step 2: trigger the 5-stage pipeline on an uploaded video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] == "processing":
        raise HTTPException(status_code=409, detail="Pipeline already running")

    video_path = jobs[job_id]["video_path"]
    jobs[job_id]["status"]   = "queued"
    jobs[job_id]["subsample"] = subsample

    background_tasks.add_task(
        run_pipeline, job_id, jobs, subsample, video_path
    )
    return {"message": "Pipeline started", "job_id": job_id}


@app.get("/status/{job_id}")
async def stream_status(job_id: str):
    """SSE stream: pushes job state to React sidebar every second."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        await asyncio.sleep(0.5)   # let background task initialize
        while True:
            yield {"data": json.dumps(jobs[job_id])}
            if jobs[job_id]["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Returns viewer-ready URLs once the pipeline is complete."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=425, detail="Pipeline not complete yet")

    heatmap_url = f"/outputs/{job_id}/heatmap.png" if os.path.exists(f"outputs/{job_id}/heatmap.png") else None
    return {
        "job_id":      job_id,
        "json_url":    f"/outputs/{job_id}/enriched_data.json",
        "ply_url":     f"/outputs/{job_id}/scene.ply",
        "log_url":     f"/outputs/{job_id}/inference_logs.txt",
        "heatmap_url": heatmap_url,
    }


@app.get("/dev/load/{job_id}")
async def dev_load(job_id: str):
    """DEV: register existing outputs on disk as a completed job — skips inference."""
    output_dir = f"outputs/{job_id}"
    json_path  = f"{output_dir}/enriched_data.json"
    ply_path   = f"{output_dir}/scene.ply"

    if not os.path.exists(json_path) or not os.path.exists(ply_path):
        raise HTTPException(
            status_code=404,
            detail=f"outputs/{job_id}/ exists but is missing enriched_data.json or scene.ply",
        )

    jobs[job_id] = {
        "status":     "completed",
        "video_path": None,
        "filename":   f"[dev] {job_id}",
        "stage":      3,
        "progress":   100,
        "logs":       [f"[dev] Loaded from disk: {output_dir}"],
    }
    heatmap_url = f"/outputs/{job_id}/heatmap.png" if os.path.exists(f"{output_dir}/heatmap.png") else None
    return {
        "job_id":      job_id,
        "json_url":    f"/outputs/{job_id}/enriched_data.json",
        "ply_url":     f"/outputs/{job_id}/scene.ply",
        "log_url":     f"/outputs/{job_id}/inference_logs.txt",
        "heatmap_url": heatmap_url,
    }


@app.get("/jobs")
async def list_jobs():
    """Debug endpoint — lists all jobs and their current status."""
    return {
        jid: {k: v for k, v in info.items() if k != "logs"}
        for jid, info in jobs.items()
    }


@app.get("/app")
async def serve_frontend():
    return FileResponse("frontend/index.html")


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Clean up outputs and job state."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] == "processing":
        raise HTTPException(status_code=409, detail="Cannot delete a running job")

    output_dir = f"outputs/{job_id}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    video_path = jobs[job_id].get("video_path")
    if video_path and os.path.exists(video_path):
        os.remove(video_path)

    del jobs[job_id]
    return {"message": f"Job {job_id} deleted"}


# Frontend static assets — must be last so API routes take priority
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")