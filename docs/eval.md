# Datasets

Our evaluate human reconstruction on 3 datasets listed below. Please download the datasets from their official sources.

- [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) 
- [EMDB](https://eth-ait.github.io/emdb/)
- [RICH](https://rich.is.tue.mpg.de/)

We follow [Multi-HMR](https://github.com/naver/multi-hmr) to prepare **3DPW** dataset, execute:

```bash
python -m eval.dataset.prepare_3dpw "create_annots()"
```
Processed 3DPW annotations will be saved to the `eval/global_human/annots/` directory.

We follow [GVHMR](https://github.com/zju3dv/GVHMR) to prepare **EMDB** dataset.

We follow [GVHMR](https://github.com/zju3dv/GVHMR) to prepare **RICH** dataset,
please download the following files and place them into the `eval/global_human/annots/RICH/` directory:
- SMPL Annotations: `hmr4d_support/rich_test_labels.pt` (via [GoogleDrive](https://drive.google.com/file/d/17fbG1IsN6DfF_KwYWR2FbNZFKvs9waVb/view?usp=drive_link))
- Camera Ground-Truth: `resource/cam2params.pt` (via [Github](https://github.com/zju3dv/GVHMR/blob/088caff492aa38c2d82cea363b78a3c65a83118f/hmr4d/dataset/rich/resource/cam2params.pt))

We also evaluate generic 3D reconstruction (camera Pose and video depth estimation), please follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to download, and follow [TTT3R](https://github.com/Inception3D/TTT3R/blob/main/eval/eval.md) to prepare **TUM-dynamics** and **Bonn** datasets.


# Evaluation

### Human Reconstruction
Results will be saved to `eval_results/global_human/*`.

```bash
# You may need to change [--num_processes] to the number of your gpus

# Local human mesh reconstruction - evaluated on 3DPW, EMDB1
CUDA_VISIBLE_DEVICES=0 bash eval/global_human/run.sh

# Global human motion estimation - evaluated on EMDB2
CUDA_VISIBLE_DEVICES=0 bash eval/global_human/run_emdb2.sh

# Global human motion estimation - evaluated on RICH
CUDA_VISIBLE_DEVICES=0 bash eval/global_human/run_rich.sh
```

### Camera Pose Estimation
Results will be saved to `eval_results/relpose/*`.

```bash
CUDA_VISIBLE_DEVICES=0 bash eval/relpose/run.sh # You may need to change [--num_processes] to the number of your gpus and choose sequence length in datasets=('tum_1000')
```

### Video Depth Estimation
Results will be saved to `eval_results/video_depth/*`.

```bash
CUDA_VISIBLE_DEVICES=0 bash eval/video_depth/run.sh # You may need to change [--num_processes] to the number of your gpus and choose sequence length in datasets=('bonn_500')
```
