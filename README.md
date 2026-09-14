# MC-SAM: Manifold-Constrained Hyper-Connections and Prompting for SAM in Camouflaged Scene Segmentation

**Public maintainer:** Yonghao Wu · [WYH302](https://github.com/WYH302)

Collaborator names and contact details are omitted from repository metadata for privacy. This identifies the public maintainer and does not assert sole authorship of the associated research.

## Abstract
Camouflaged object detection (COD) seeks to segment targets that share near-identical appearance with their surroundings---a task that pushes the limits of visual recognition. The Segment Anything Model (SAM) offers class-agnostic generalization, yet its reliance on manually specified prompts, coupled with a lack of task-specific adaptation, makes it fragile on COD benchmarks where prompt ambiguity is the norm. We propose MC-SAM, an adaptive framework that tackles these limitations through three tightly coupled innovations: (1) Hyperparameter Conditioning (HyperCond), a learned module that converts fixed thresholds and loss weights into sample-adaptive variables through Beta-distribution sampling; (2) a Manifold-Constrained Adapter whose mixing weights are projected onto the Birkhoff polytope via Sinkhorn--Knopp iterations, guaranteeing energy-bounded fusion and cross-layer stability; and (3) a Cross-Space Stable Prompt Generator that regularizes the textual--visual alignment through manifold constraints, suppressing covariance ill-conditioning. A complementary RankDice-RMA post-processing step further sharpens boundaries by analytically maximizing the expected Dice score, removing the need for manual prompt tuning. On four COD benchmarks (COD10K, NC4K, CAMO, CHAMELEON), MC-SAM attains structure-measure scores of 0.903, 0.896, 0.872, and 0.930, respectively---matching or surpassing state-of-the-art methods while training only a fraction of their parameters.

## Keywords
Camouflaged Object Detection; Segment Anything Model (SAM); Manifold Constraints; Hyperparameter Conditioning; RankDice-RMA

## Quick Start

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Usage
- **Training:** Run `python MCsam_train.py`
- **Inference:** Run `python inference_mcsam.py`

## Citation
Use the author-approved manuscript for the paper's bibliographic citation. The maintainer entry above is not a substitute for its author list. To identify this software version, include the repository URL and the commit used.
