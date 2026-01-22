# How to Start
> This repo is an **end-to-end** builder for the dataset described in your build spec.
>
> ## Quick start
>
> ```bash
> bash run.sh
> ```
>
> By default, it will:
>
> 1. Create & activate an environment (conda if available, else python venv)
> 2. Install **PyTorch**, **torchvision**, **detectron2**
> 3. Download **MS COCO 2017** train/val + annotations
>    - Default: download from a **Hugging Face mirror** repo (often faster than the official COCO host)
>    - Fallback: can use the official COCO URLs
> 4. Download the Detectron2 model zoo checkpoint for:
>    `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`
> 5. Build the dataset into:
>
> ```
> coco-spatial-1b/
>   meta/
>     build_manifest.json
>     stats.json
>   data/
>     images.parquet
>     rects/
>       train2017/
>         shard-000000.parquet
>         shard-001024.parquet
>         ...
>       val2017/
>         shard-118287.parquet
>         ...
> ```
>
> ## Config overrides (env vars)
>
> Common overrides:
>
> ```bash
> # Put COCO somewhere else:
> COCO_ROOT=/data/coco2017
> 
> # Output directory:
> OUTPUT_DIR=/data/coco-spatial-1b
> 
> # Force CPU (not recommended for full build):
> DEVICE=cpu TORCH_CUDA=cpu
> 
> # Force a specific CUDA wheel flavor:
> TORCH_CUDA=cu121
> 
> # Pin detectron2 to a specific git commit:
> DETECTRON2_GIT_REF=<commit_hash>
> 
> # Root seed:
> SEED_ROOT=12345
> 
> bash run.sh
> 
> # Use official COCO host instead of Hugging Face mirror
> COCO_DOWNLOAD_SOURCE=official bash run.sh
> 
> # Use a different HF mirror repo (must contain train2017.zip / val2017.zip / annotations_trainval2017.zip)
> HF_COCO_REPO=pcuenq/coco-2017-mirror COCO_DOWNLOAD_SOURCE=hf bash run.sh
> 
> # Disable HF transfer accelerators (enabled by default for HF downloads)
> ENABLE_HF_TRANSFER=0 ENABLE_HF_XET_HIGH_PERFORMANCE=0 bash run.sh
> ```
>
> ## Notes
>
> - This build writes **~1.23B proposals**, so it requires **lots** of disk.
> - If `detectron2` compilation fails, install build tools (`gcc/g++`, `cmake`) and (for CUDA) a CUDA toolkit with `nvcc`.

# Building Spec.

> This part defines the **data sources**, **geometric semantics**, **generation pipeline**, **output format**, and **acceptance rules** for **COCO‑Spatial-Join‑1B**. The build implementation **must** satisfy every item in this specification. The keywords “must / must not / should” are used as **mandatory constraints**.
>
> ------
>
> ## 0. Dataset Overview
>
> ### 0.1 Data Sources
>
> - Images and annotations: **MS COCO 2017** `train2017` and `val2017`
>   - `train2017`: 118,287 images
>   - `val2017`: 5,000 images
> - Only the following COCO information is used:
>   - Image metadata: `width`, `height`, `file_name`, `id` (COCO `image_id`)
>   - Detection annotations: `bbox`, `iscrowd`, `category_id`, `id` (COCO `ann_id`)
> - Proposal source: run **Detectron2** **Faster R‑CNN (ResNet‑50‑FPN)** on each image and take the **RPN** output proposals.
>
> ### 0.2 Generated Objects
>
> For each image, generate two types of axis-aligned rectangular objects, and represent them uniformly as a 3D half-open box:
>
> 1. **GT (Ground Truth)**: from COCO detection annotations, **including all annotations with `iscrowd ∈ {0,1}`** (no filtering).
> 2. **PROPOSAL (RPN proposals)**: from the RPN output, **exactly 10,000 proposals per image**.
>
> ### 0.3 Scale
>
> - Total number of images: `N = 118,287 + 5,000 = 123,287`
> - Total number of proposals: `N × 10,000 = 1,232,870,000` (about 1.233 billion)
> - Total number of GT boxes: equal to the number of COCO annotation bboxes (including crowd)
>
> ------
>
> ## 1. Geometric Semantics (Unified and Fixed)
>
> ### 1.1 Unified Object Representation: 3D Axis-Aligned Half-Open Box
>
> Each object is represented as a 3D half-open box:
> $$
> b=[x_{\min},x_{\max})\times[y_{\min},y_{\max})\times[z_{\min},z_{\max})
> $$
> and must satisfy:
>
> - $x_{\min }<x_{\max },\ y_{\min }<y_{\max },\ z_{\min }<z_{\max }$
> - Coordinate system: **original COCO image pixel coordinate system** (based on the original image size)
> - Coordinate precision: `min_x/min_y/max_x/max_y` must be **float32**
>
> ### 1.2 z-Dimension Construction: Slice by Image
>
> Each image is assigned a consecutive integer `z_idx ∈ {0,…,N-1}`, and it is enforced that:
>
> - `z_min = z_idx`
> - `z_max = z_idx + 1`
>
> That is, every object's z-interval is **[z_idx, z_idx+1)**, and different images **strictly do not overlap** along the z-dimension.
>
> ### 1.3 Intersection Test: Half-Open Semantics (Touching Boundaries Does Not Count as Intersecting)
>
> Two 3D boxes $b,b'$ intersect if and only if they strictly overlap on every dimension:
> $$
> \max(L_i(b),L_i(b'))<\min(R_i(b),R_i(b')),\ \ i\in\{x,y,z\}
> $$
>
> ------
>
> ## 2. Benchmark Task Definitions (Fixed)
>
> ### 2.1 Task A: GT × PROPOSAL
>
> - Relation $R$: all objects with `type=GT`
> - Relation $S$: all objects with `type=PROPOSAL`
> - Join predicate: 3D `intersects` (per 1.3)
> - Output: ordered pairs $(r,s)$ where $r\in R$, $s\in S$
>
> ### 2.2 Task B: PROPOSAL × PROPOSAL (Split into Two Halves)
>
> For each image, proposals are split into two halves by `rank` (see 4.6):
>
> - Relation $R$: `type=PROPOSAL` and `rank ∈ [1,5000]`
> - Relation $S$: `type=PROPOSAL` and `rank ∈ [5001,10000]`
> - Join predicate: 3D `intersects` (per 1.3)
> - Output: ordered pairs $(r,s)$ where $r\in R$, $s\in S$
>
> ------
>
> ## 3. Input Files (COCO)
>
> The following must be available during building:
>
> - `instances_train2017.json`
> - `instances_val2017.json`
> - Image directories:
>   - `train2017/`
>   - `val2017/`
>
> The JSON files must contain and use the following fields:
>
> - `images`: `id`, `file_name`, `width`, `height`
> - `annotations`: `id` (ann_id), `image_id`, `bbox`, `category_id`, `iscrowd`
> - `categories`: `id` (category_id) and name (only for aligning COCO semantics; this dataset does not depend on category names)
>
> ------
>
> ## 4. z_idx Mapping Rules (Fixed and Reproducible)
>
> ### 4.1 Full Image Set and Ordering
>
> Merge image metadata from `train2017` and `val2017` into the full set `ImagesAll`, and sort by the following keys:
>
> 1. `split`: `train2017` first, `val2017` second
> 2. `coco_image_id`: ascending
>
> ### 4.2 z_idx Assignment
>
> Assign `z_idx` in the sorted order:
>
> - For the i-th image, `z_idx = i` (starting from 0)
> - `z_min = z_idx`
> - `z_max = z_idx + 1`
>
> ------
>
> ## 5. Model and Proposal Generation (10,000 per Image)
>
> ### 5.1 Framework and Model
>
> - Framework: Detectron2 (PyTorch)
> - Model: Faster R‑CNN + ResNet‑50‑FPN
> - Proposal source: output of the **RPN proposal generator**, not the final ROI Heads detections.
>
> RPN output is represented as `Instances`, and the following must be read:
>
> - `"proposal_boxes"`
> - `"objectness_logits"` (logit)
>
> ### 5.2 Deterministic Execution (Mandatory)
>
> The build process must enable deterministic settings during inference and record the following in the artifacts:
>
> - `seed` (a single integer seed as the root seed for the whole pipeline)
> - Seed derivation rules for Python/NumPy/PyTorch (must be written into the manifest)
> - Inference must use:
>   - `model.eval()`
>   - `torch.inference_mode()` (or an equivalent no-gradient inference mode)
> - Switches that affect numerical reproducibility must be disabled (must include at least):
>   - `torch.backends.cudnn.benchmark = False`
>   - `torch.use_deterministic_algorithms(True)` (or equivalent)
>   - TF32 must be disabled: `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`
> - Data reading order must be determined by the total order of `z_idx`. Parallel processing is allowed, but writing must follow the deterministic order in 8.2.
>
> ### 5.3 Weights and Version Recording (Must Be Persisted)
>
> The build artifacts must record:
>
> - Detectron2 version (pip version and git commit if available)
> - PyTorch version, CUDA version, cuDNN version
> - Model config identifier (local path or an equivalently resolvable identifier)
> - Checkpoint source URL
> - Checkpoint file SHA256
>
> ### 5.4 Inference Preprocessing (Fixed)
>
> For each image, the following transforms consistent with Detectron2 standard inference must be applied before feeding into the model:
>
> - `ResizeShortestEdge`:
>   - resize the short edge to 800
>   - long edge not exceeding 1333
>   - keep aspect ratio
> - Use Detectron2 standard normalization and tensor layout
>
> **Coordinate remapping must happen before writing:**
> RPN outputs proposals in the transformed image coordinate system. The final written `min_x/min_y/max_x/max_y` must be mapped back to the **original image coordinate system**, and clipped to $[0,width]\times[0,height]$.
>
> ### 5.5 RPN Configuration (Fixed to Produce Sufficient Candidates)
>
> This dataset targets “high overlap pressure testing.” The RPN must provide a sufficiently large candidate pool, then be truncated offline to 10,000 proposals per image.
>
> The test/inference branch must use:
>
> - `MODEL.RPN.PRE_NMS_TOPK_TEST = 20000`
> - `MODEL.RPN.POST_NMS_TOPK_TEST = 20000`
> - `MODEL.RPN.NMS_THRESH = 1.0`
>
> And it is required that:
>
> - The build implementation must ensure **NMS will not remove any boxes** (the threshold configuration and implementation semantics must satisfy this), and verify on each image that the candidate pool size is `>= 10000`.
> - The final proposals must be produced by the sorting and truncation rules in 5.6, ensuring **exactly** 10,000 outputs per image.
>
> ### 5.6 Score Definition and Sorting (Fixed)
>
> - RPN returns `objectness_logits` (real-valued logits)
>
> - The stored `score` is defined as:
>   $$
>   \text{score}=\sigma(\text{objectness\_logit})=\frac{1}{1+\exp(-\text{logit})}
>   $$
>   and stored as **float32**, with value range $[0,1]$.
>
> #### 5.6.1 Sorting Key (Deterministic Total Order)
>
> For each image, candidate proposals must first complete the coordinate normalization in Chapter 6 (including float32 casting and degenerate-box normalization), then be sorted and truncated.
>
> The sorting key is lexicographic (from higher priority to lower):
>
> 1. `score` **descending**
> 2. `(min_x, min_y, max_x, max_y)` **ascending** (4-tuple lexicographic order; all values are float32)
> 3. `cand_idx` **ascending** (0-based index in the candidate list, used to make the ordering a strict total order)
>
> After sorting, take the top 10,000 and define:
>
> - `rank = 1..10000` (consistent with the ordering)
> - `type = PROPOSAL`
>
> ------
>
> ## 6. Coordinate Normalization (Must Hold Before Writing)
>
> Any GT or PROPOSAL must satisfy the following before being written:
>
> - `0 ≤ min_x < max_x ≤ width`
> - `0 ≤ min_y < max_y ≤ height`
> - `z_max = z_min + 1`
> - `min_x/min_y/max_x/max_y/score` must all be **finite float32** (no NaN/Inf allowed)
>
> ### 6.1 Bounding Rule (Clip)
>
> First, bound x/y coordinates (effective under float32):
>
> - `min_x, max_x` clip to `[0, width]`
> - `min_y, max_y` clip to `[0, height]`
>
> The handling rules for non-finite values are fixed as follows (performed before clipping):
>
> - `NaN → 0.0`
> - `+Inf → upper bound` (`width` for x, `height` for y)
> - `-Inf → 0.0`
>
> ### 6.2 Degenerate Box Normalization (max ≤ min)
>
> If `max_x ≤ min_x` or `max_y ≤ min_y` occurs, deterministic handling is required, and the result must remain within bounds.
>
> The following rules must be executed in **float32 precision**:
>
> - If `max_x ≤ min_x`:
>   1. Set `max_x = nextafter(min_x, +∞)`
>   2. If `max_x > width`, set `max_x = width` and `min_x = nextafter(width, -∞)`
> - Apply the same logic to y (using `height` as the boundary)
>
> ------
>
> ## 7. Output Directory Structure (Fixed)
>
> The generation output uses Parquet shards (columnar compression), together with metadata files. The directory structure is fixed as:
>
> ```text
> coco-spatial-1b/
>   meta/
>     build_manifest.json
>     stats.json
>   data/
>     images.parquet
>     rects/
>       train2017/
>         shard-000000.parquet
>         shard-001024.parquet
>         ...
>       val2017/
>         shard-118287.parquet
>         ...
> ```
>
> ------
>
> ## 8. Sharding Rules (Fixed, No Cross-Split Mixing)
>
> ### 8.1 Shard Coverage Range
>
> - Each shard covers a **continuous segment of images by z_idx**, with length at most 1024 images.
> - `train2017` and `val2017` must be written into their respective directories, and no shard file may contain images from both splits.
>
> Define:
>
> - `TRAIN_Z = [0, 118287)`
> - `VAL_Z   = [118287, 123287)`
>
> Therefore:
>
> - Shards under `train2017/` may only cover z_idx in `TRAIN_Z`
> - Shards under `val2017/` may only cover z_idx in `VAL_Z`
>
> ### 8.2 Shard File Name and Content Range
>
> Shard file names are based on the **starting z_idx**:
>
> - File name format: `shard-{start_z_idx:06d}.parquet`
>
> - Each shard covers:
>   $$
>   z\_idx \in [start\_z\_idx,\ \min(start\_z\_idx+1024,\ split\_end))
>   $$
>   where `split_end` is the upper bound of the split (118287 for train, 123287 for val).
>
> ### 8.3 Row Order Within a Shard (Deterministic)
>
> Each `rects` shard must be written with rows sorted by `rect_id` ascending.
>
> `images.parquet` must be written in ascending `z_idx` order.
>
> ------
>
> ## 9. Parquet Schema (Fixed)
>
> ### 9.1 `images.parquet`
>
> One row per image, fields and types:
>
> - `split` : int8 (0=train2017, 1=val2017)
> - `coco_image_id` : int32
> - `file_name` : string
> - `width` : int32
> - `height` : int32
> - `z_idx` : int32
>
> ### 9.2 `rects` (One 3D Box per Row)
>
> Fields and types (fixed):
>
> - `rect_id` : int64
> - `z_min` : int32
> - `z_max` : int32
> - `min_x` : float32
> - `min_y` : float32
> - `max_x` : float32
> - `max_y` : float32
> - `type` : int8 (0=GT, 1=PROPOSAL)
> - `rank` : int16 (GT=0; PROPOSAL=1..10000)
> - `score` : float32 (GT=1.0; PROPOSAL=sigmoid(objectness_logit))
> - `category_id` : int16 (GT=COCO category_id; PROPOSAL=-1)
> - `iscrowd` : int8 (GT=COCO iscrowd; PROPOSAL=0)
> - `coco_ann_id` : int64 (GT=COCO annotation id; PROPOSAL=-1)
> - `coco_image_id` : int32
>
> ------
>
> ## 10. rect_id Generation Rules (Fixed, No Overflow, Invertible)
>
> ### 10.1 Basic Constant
>
> - `STRIDE = 20000` (fixed)
>
> ### 10.2 Per-Image Base (Must Use int64)
>
> For each image (given `z_idx`):
>
> - `base = int64(z_idx) * int64(STRIDE)`
>
> No implementation may compute `base` or related intermediates in int32.
>
> ### 10.3 GT rect_id
>
> - Take all GT annotations for the image, sort by `coco_ann_id` ascending
> - Obtain local indices `gt_local_idx = 0..num_gt-1`
> - `rect_id = base + gt_local_idx`
>
> And require:
>
> - `num_gt ≤ 10000` (must be validated)
>
> ### 10.4 PROPOSAL rect_id
>
> - `rect_id = base + 10000 + (rank - 1)`
>
> ------
>
> ## 11. Build Pipeline (Normalized Workflow)
>
> The build implementation must follow the stage order below (internal parallelism is allowed, but the output must be logically equivalent):
>
> 1. Parse COCO `instances_train2017.json` and `instances_val2017.json`, generate `ImagesAll`, sort per 4.1 and assign `z_idx`.
> 2. Write `data/images.parquet` (ascending by `z_idx`).
> 3. Generate GT per image:
>    - Convert COCO `bbox=[x,y,w,h]` into `(min_x,min_y,max_x,max_y)=(x,y,x+w,y+h)`
>    - Apply coordinate normalization in Chapter 6
>    - Write z dimension: `z_min=z_idx, z_max=z_idx+1`
> 4. Generate proposals per image:
>    - Apply inference preprocessing in 5.4, run the model to obtain the RPN candidate pool (must verify size `>=10000`)
>    - Map coordinates back to the original image coordinate system and clip to image boundaries
>    - Apply coordinate normalization in Chapter 6
>    - `score = sigmoid(objectness_logit)` (float32)
>    - Sort by the key in 5.6 and take the top 10,000; assign `rank=1..10000`
> 5. Write all shards under `rects/train2017/` and `rects/val2017/` per Chapter 8:
>    - Within each shard, rows are written in ascending `rect_id`
> 6. Generate `meta/build_manifest.json` and `meta/stats.json` (see Chapter 12)
>
> ------
>
> ## 12. Metadata (Must Be Generated)
>
> ### 12.1 `meta/build_manifest.json`
>
> Must include the following fields (field names may use JSON snake_case):
>
> - **COCO inputs**
>   - `coco_train_json`: file name
>   - `coco_val_json`: file name
>   - `coco_train_json_sha256`
>   - `coco_val_json_sha256`
> - **Build software environment**
>   - `builder_version` (version or git commit of the build script itself)
>   - `python_version`
>   - `numpy_version`
>   - `pyarrow_version` (or equivalent parquet writer information)
>   - `detectron2_version` (pip version and git commit)
>   - `torch_version`
>   - `cuda_version`
>   - `cudnn_version`
> - **Determinism settings**
>   - `seed_root`
>   - `seed_rules` (describe how Python/NumPy/Torch seeds are derived)
>   - `determinism_flags` (cudnn_benchmark, use_deterministic_algorithms, tf32, etc.)
> - **Model information**
>   - `model_config_id` (resolvable identifier)
>   - `rpn_pre_nms_topk_test`
>   - `rpn_post_nms_topk_test`
>   - `rpn_nms_thresh`
>   - `checkpoint_url`
>   - `checkpoint_sha256`
> - **Build information**
>   - `build_time_utc`
>   - `host_os`
>   - `cpu_model`
>   - `gpu_model` (and count)
>   - `ram_gb`
> - **Output inventory**
>   - `num_images_total` (should be 123287)
>   - `num_proposals_total` (should be 1232870000)
>   - `shards`: for each shard, `{split, shard_file, start_z_idx, end_z_idx, num_rows}`
>
> ### 12.2 `meta/stats.json`
>
> Must include:
>
> #### 12.2.1 Per-image Statistics
>
> - For each `z_idx`:
>   - `num_gt_total`
>   - `num_gt_crowd`
>
> (The storage format may be an array in `z_idx` order.)
>
> #### 12.2.2 Proposal Statistics Histograms
>
> Must provide both `bin_edges` and `counts`, with fixed bin definitions as follows:
>
> - **Score histogram**
>   - `bin_edges`: `[0.000, 0.001, 0.002, ..., 1.000]` (1001 edges, 1000 bins)
>   - `counts[i]`: number of proposals with `score ∈ [edges[i], edges[i+1])` (the last bin may be `[0.999, 1.000]` including the right endpoint)
> - **Area (pixel area) histogram**
>   - Definition: `area = (max_x-min_x) * (max_y-min_y)` (computed in float64, unit: px²)
>   - `bin_edges`: `[0, 1, 2, 4, 8, ..., 2^24]`
>   - Extra field: `overflow_count_ge_2p24` (count of area ≥ 2^24)
> - **Aspect ratio histogram**
>   - Definition: `ar = (max_x-min_x) / (max_y-min_y)` (float64)
>   - `bin_edges`: `[0] + [2^k for k=-8..8] + [2^9]`
>   - Extra fields:
>     - `underflow_count_lt_2m8` (count of ar < 2^-8)
>     - `overflow_count_ge_2p9` (count of ar ≥ 2^9)
>
> #### 12.2.3 GT Statistics
>
> - **GT area histogram**: identical area bins to the proposals' area bins
> - **iscrowd ratio**
>   - `gt_iscrowd_ratio = total_crowd_gt / total_gt`
>
> ------
>
> ## 13. Quality Acceptance (All Must Pass)
>
> ### 13.1 Per-image Acceptance (For Every z_idx)
>
> For each image, all of the following must hold:
>
> 1. Number of PROPOSAL rows = 10,000
> 2. PROPOSAL `rank` is exactly 1..10000
> 3. PROPOSAL `score` is non-increasing with `rank`
>    - Strict definition: for rank i<j, must have `score[i] ≥ score[j]`
> 4. If `score` is equal, ordering must be consistent with the sorting rule `(min_x, min_y, max_x, max_y, cand_idx)` (the build implementation must be able to validate this)
> 5. GT has `rank=0` and `score=1.0` (exactly representable float32)
> 6. All objects satisfy the coordinate normalization constraints in Chapter 6:
>    - bounded, strict half-open inequalities hold
>    - coordinates and scores are finite float32
> 7. All objects satisfy `z_max = z_min + 1`
>
> Any failure is considered a build failure.
>
> ### 13.2 Global Acceptance
>
> The following must hold and the verification results must be output:
>
> - Total proposals = `123,287 × 10,000 = 1,232,870,000`
> - Number of rows in `images.parquet` = 123,287
> - `z_idx` covers `[0, N-1]` with no missing values and no duplicates
> - For each shard:
>   - covered z_idx are continuous and match the file name
>   - no cross-split mixing
>   - within the shard, `rect_id` is strictly increasing
> - No `rect_id` collisions across the full dataset (per-image invertibility validation is a sufficient condition)
>
> ------
>
> ## 14. Field Semantic Conventions (For Downstream Consumption)
>
> - `type=GT`:
>   - `rank=0`
>   - `score=1.0`
>   - `category_id` and `iscrowd` come from COCO
>   - `coco_ann_id` is the COCO annotation `id`
> - `type=PROPOSAL`:
>   - `rank=1..10000`
>   - `score=sigmoid(objectness_logit)`
>   - `category_id=-1`
>   - `iscrowd=0`
>   - `coco_ann_id=-1`
>

