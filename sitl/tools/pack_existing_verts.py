#!/usr/bin/env python3
"""Convert a job's inline-JSON verts into a Float32 binary side-car.

A long clip's dashboard_data.json (10475 verts/human/frame) exceeds Chrome's
~512 MB max JS-string length, so the browser cannot JSON.parse it and the SMPL-X
mesh silently falls back to a marker sphere. This rewrites such a job to match the
current engine.py output:

    verts.bin          concatenated Float32LE, one [V,3] block per human-instance
    verts_index.json   {dtype, vert_count, stride, frames: [[block,...], ...]}
    dashboard_data.json  same file with "verts" stripped from each human (slim)

Usage:
    python sitl/tools/pack_existing_verts.py <job_id> [<job_id> ...]
    python sitl/tools/pack_existing_verts.py --all      # every job under outputs/
"""
import json
import os
import sys

import numpy as np

OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")


def pack_job(job_id):
    d = os.path.join(OUTPUTS, job_id)
    json_path = os.path.join(d, "dashboard_data.json")
    if not os.path.exists(json_path):
        print(f"[skip] {job_id}: no dashboard_data.json")
        return

    with open(json_path) as f:
        data = json.load(f)

    blocks = []
    index = {"frames": []}
    vert_count = None
    had_inline = False

    for frame in data.get("frames", []):
        frame_blocks = []
        for h in frame.get("humans", []):
            v = h.pop("verts", None)
            if v is None:
                continue
            had_inline = True
            arr = np.asarray(v, dtype=np.float32)
            if vert_count is None:
                vert_count = arr.shape[0]
            frame_blocks.append(len(blocks))
            blocks.append(arr.reshape(-1))
        index["frames"].append(frame_blocks)

    if not had_inline:
        print(f"[skip] {job_id}: no inline verts to pack")
        return

    bin_path = os.path.join(d, "verts.bin")
    np.concatenate(blocks).astype("<f4").tofile(bin_path)
    index.update({"dtype": "float32", "vert_count": int(vert_count),
                  "stride": int(vert_count) * 3})
    with open(os.path.join(d, "verts_index.json"), "w") as f:
        json.dump(index, f)
    with open(json_path, "w") as f:        # slim dashboard (verts stripped)
        json.dump(data, f)

    print(f"[ok]   {job_id}: {len(blocks)} blocks → "
          f"verts.bin {os.path.getsize(bin_path) / 1048576:.0f} MB, "
          f"dashboard now {os.path.getsize(json_path) / 1048576:.0f} MB")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    if args == ["--all"]:
        args = sorted(n for n in os.listdir(OUTPUTS)
                      if os.path.isdir(os.path.join(OUTPUTS, n)))
    for job_id in args:
        pack_job(job_id)


if __name__ == "__main__":
    main()
