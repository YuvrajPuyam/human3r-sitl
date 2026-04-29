import asyncio
from .workers.inference import run_human3r_inference
from .workers.analytics import compute_spatial_analytics


async def run_pipeline(job_id: str, jobs_store: dict, subsample: int, video_path: str):
    try:
        # Stage 1 — Human3R inference (engine.py subprocess)
        jobs_store[job_id].update({"status": "processing", "stage": 1})
        jobs_store[job_id]["logs"].append("Launching Human3R inference engine...")
        await run_human3r_inference(job_id, subsample, video_path, jobs_store)

        # Stage 2 — Spatial analytics
        jobs_store[job_id].update({"stage": 2, "progress": 0})
        jobs_store[job_id]["logs"].append("Computing spatial analytics...")
        await compute_spatial_analytics(job_id)
        jobs_store[job_id]["logs"].append("enriched_data.json written.")

        # Stage 3 — Done
        jobs_store[job_id].update({"stage": 3, "status": "completed", "progress": 100})
        jobs_store[job_id]["logs"].append("Pipeline complete. Viewer ready.")

    except Exception as e:
        jobs_store[job_id].update({"status": "failed"})
        jobs_store[job_id]["logs"].append(f"Pipeline error: {e}")
        raise
