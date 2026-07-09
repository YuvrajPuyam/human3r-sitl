import asyncio
from .workers.inference import run_human3r_inference
from .workers.analytics import compute_spatial_analytics
from .state import persist_jobs

# Only one Human3R inference may touch the GPU at a time. A second /run request
# is admitted but parks here as "queued" until the first releases the GPU,
# instead of racing for VRAM and OOM-killing both. Analytics (CPU-bound) runs
# outside the guard so it doesn't hold the GPU slot.
_INFERENCE_SEM = asyncio.Semaphore(1)


async def run_pipeline(job_id: str, jobs_store: dict, subsample: int, video_path: str):
    try:
        # Wait for the GPU slot (stays "queued" while another job holds it).
        jobs_store[job_id].update({"status": "queued", "stage": 1})
        if _INFERENCE_SEM.locked():
            jobs_store[job_id]["logs"].append("Waiting for GPU (another job is running)...")
        persist_jobs()

        async with _INFERENCE_SEM:
            # Stage 1 — Human3R inference (engine.py subprocess)
            jobs_store[job_id].update({"status": "processing", "stage": 1})
            jobs_store[job_id]["logs"].append("Launching Human3R inference engine...")
            persist_jobs()
            await run_human3r_inference(job_id, subsample, video_path, jobs_store)

        # Stage 2 — Spatial analytics (CPU-bound; runs outside the GPU guard)
        jobs_store[job_id].update({"stage": 2, "progress": 0})
        jobs_store[job_id]["logs"].append("Computing spatial analytics...")
        await compute_spatial_analytics(job_id)
        jobs_store[job_id]["logs"].append("enriched_data.json written.")

        # Stage 3 — Done
        jobs_store[job_id].update({"stage": 3, "status": "completed", "progress": 100})
        jobs_store[job_id]["logs"].append("Pipeline complete. Viewer ready.")
        persist_jobs()

    except Exception as e:
        jobs_store[job_id].update({"status": "failed"})
        jobs_store[job_id]["logs"].append(f"Pipeline error: {e}")
        persist_jobs()
        raise
