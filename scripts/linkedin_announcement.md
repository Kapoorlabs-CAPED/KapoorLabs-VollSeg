# LinkedIn — 3D StarDist · Keras → PyTorch Lightning announcement

---

**KapoorLabs has just ported 3D StarDist from TensorFlow/Keras to PyTorch Lightning — line by line — and we're releasing the whole thing free for the world.** 🧬

3D StarDist (Weigert, Schmidt et al., MICCAI 2020) is the de-facto standard for nucleus instance segmentation in volumetric microscopy. For five years it has lived in TensorFlow/Keras, with an OpenMP C++ kernel for the polyhedron rasteriser. Outstanding science, but increasingly painful to maintain alongside modern PyTorch training stacks.

Over the last few months our team has rebuilt 3D StarDist end-to-end in PyTorch Lightning, matching the upstream algorithm at the file level — not "inspired by", but a faithful port that produces results indistinguishable from the original on real microscopy data.

What's in the box:

✅ **Word-by-word algorithmic port** of `predict_instances` → label image: rays + faces (golden spiral + ConvexHull), CSBDeep tile iterator (verbatim), kernel + convex-hull short-circuit polyhedron rasteriser (the exact two-tier test from `stardist3d_impl.cpp`), score-descending NMS, "earlier-cell-wins" paint rule.

✅ **Lightning-native training**: `careamics` U-Net backbone, distance + probability heads, anisotropy-aware ray geometry, Hydra-driven configs, SLURM-friendly sweep + threshold-optimisation scripts.

✅ **Hugging Face Hub integration**: pretrained Xenopus models live at `KapoorLabs/xenopus-stardist-pytorch` (+ U-Net, MaskUNet, CARE companions). One line — `ensure_model(...)` — and you're predicting.

✅ **Production-ready ROI-gated pipeline**: wraps StarDist in a Mask-UNet bbox crop that fixes percentile-saturation artefacts on near-empty timepoints. Validated against the legacy keras reference across early/mid/late embryo stages — mean volume, mean radius and total tissue surface area all within ~2 % of the keras reference at every developmental stage.

✅ **Vendored upstream C++ sources** in `_lib/` (BSD-3) so the algorithmic source-of-truth lives next to our Python port. Build the native kernel optionally for full C++ throughput.

✅ **Pure-numpy fallback** that matches the C++ output bit-for-bit on the algorithms that matter (rasteriser, NMS, paint) — zero runtime dependency on the upstream `stardist` package or on `tensorflow`.

🎁 **Free.** BSD-3 licensed, in `kapoorlabs-vollseg`:

`pip install kapoorlabs-vollseg`

Source: https://github.com/Kapoorlabs-CAPED/KapoorLabs-VollSeg

A massive thank-you to Uwe Schmidt and Martin Weigert for the original StarDist work and for keeping the upstream BSD-3 — it's what made this whole port legally and ethically possible. We hope this Lightning-native reimplementation broadens the audience for 3D StarDist into the PyTorch ecosystem and unlocks the next wave of microscopy-AI engineering.

Onward 🚀

#OpenSource #DeepLearning #Microscopy #PyTorch #PyTorchLightning #ComputerVision #BioImage #BioImageAnalysis #StarDist #VollSeg #KapoorLabs

---

## Hashtag bank (use top 5-7)

#OpenSource #DeepLearning #PyTorch #PyTorchLightning #Microscopy #BioImageAnalysis #BioImage #ComputerVision #ImageSegmentation #InstanceSegmentation #StarDist #VollSeg #HuggingFace #SegmentationModels #Cellpose #3DImaging #LifeSciences #AI4Bio #KapoorLabs #Reproducibility #ScientificSoftware

## Suggested attachment

The early/mid/late distribution box-plot from `compare_roi_stardist_vs_keras.ipynb` — the side-by-side panel that shows mean volume / radius / surface area landing on top of the keras reference at every developmental stage. Visually settles the "did you actually validate it?" question in two seconds.

## Optional shorter version (1 200 chars-ish for a single-post feed)

**KapoorLabs has ported 3D StarDist (Weigert, Schmidt et al.) from TensorFlow/Keras to PyTorch Lightning — line by line — and we're releasing it free under BSD-3.**

The full algorithm: rays + ConvexHull faces, CSBDeep tiling, kernel + convex-hull short-circuit polyhedron rasteriser, score-descending NMS, earlier-cell-wins paint — all reimplemented natively. Lightning-native training, Hydra configs, HuggingFace Hub for pretrained models, ROI-gated production pipeline.

Validated against the legacy keras reference on Xenopus embryos: mean volume, radius and surface area within ~2 % of keras at every developmental stage. Zero runtime dependency on the upstream `stardist` package or `tensorflow`.

`pip install kapoorlabs-vollseg`
Repo: https://github.com/Kapoorlabs-CAPED/KapoorLabs-VollSeg

Massive thanks to the original StarDist authors for the science and for the BSD-3 license that made this port possible.

#OpenSource #PyTorch #DeepLearning #Microscopy #BioImageAnalysis #StarDist #KapoorLabs
