"""End-to-end PyTorch StarDist demo: H5 prep → Lightning training → prediction.

Three subcommands, run in order:

    # 1. Generate H5 of (raw, label) patches.  Rays are NOT computed here —
    #    they're a property of training, not data prep.
    python scripts/02_train_stardist_pytorch.py prep \\
        --raw  data/raw \\
        --label data/labels \\
        --out  out/stardist_train.h5 \\
        --patch  16 256 256

    # 2. Train (Lightning saves checkpoints under <model_dir>/<model_name>/
    #    plus a sidecar <model_name>.rays.npy).
    python scripts/02_train_stardist_pytorch.py train \\
        --h5         out/stardist_train.h5 \\
        --model-dir  out/models \\
        --model-name xenopus_v1 \\
        --n-rays 96 --anisotropy 2.0 1.0 1.0 \\
        --epochs 100 --batch-size 4 --lr 4e-4 \\
        --augment

    # 3. Predict on a single volume → uint16 label image.
    python scripts/02_train_stardist_pytorch.py predict \\
        --ckpt   out/models/xenopus_v1/last.ckpt \\
        --rays   out/models/xenopus_v1.rays.npy \\
        --image  data/test/some_volume.tif \\
        --out    out/predictions/some_volume_seg.tif \\
        --prob-thresh 0.5 --nms-thresh 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite
from torch.utils.data import DataLoader

from vollseg import StarDistSegmenter, StarDistTrainer
from vollseg.stardist import (
    Compose,
    InputGaussianNoise,
    InputPercentileNormalize,
    RandomFlip,
    RandomRot90,
    StarDistH5Dataset,
    generate_stardist_h5,
    rays_2d,
    rays_3d_golden_spiral,
    stardist_collate,
)


# ============================================================== prep

def cmd_prep(args):
    if len(args.patch) not in (2, 3):
        raise SystemExit(f"--patch must have 2 or 3 values, got {args.patch}")
    counts = generate_stardist_h5(
        raw_dir=args.raw,
        label_dir=args.label,
        output_h5=args.out,
        patch_shape=tuple(args.patch),
        val_files=args.val_files,
        min_foreground_ratio=args.min_fg,
        overwrite=args.overwrite,
    )
    print(f"Done: {counts}")


# ============================================================== train

def _build_rays(args, ndim: int) -> np.ndarray:
    if ndim == 2:
        return rays_2d(args.n_rays)
    anisotropy = tuple(args.anisotropy) if args.anisotropy else None
    return rays_3d_golden_spiral(args.n_rays, anisotropy=anisotropy)


def _build_train_transform(pmin: float, pmax: float, augment: bool, gauss_std: float):
    transforms = [InputPercentileNormalize(pmin=pmin, pmax=pmax)]
    if augment:
        # Geometric augmentation works in any ndim — the dataset re-derives
        # the (prob, dist) targets from the augmented label so no ray-channel
        # gymnastics are needed.
        transforms.extend([RandomFlip(p=0.5), RandomRot90(p=0.5)])
        if gauss_std > 0:
            transforms.append(InputGaussianNoise(std=gauss_std, p=0.3))
    return Compose(transforms)


def cmd_train(args):
    # Inspect the H5 to discover patch shape → infer ndim → build rays.
    import h5py
    with h5py.File(args.h5, "r") as f:
        patch_shape = f["train"]["raw"].shape[1:]
    ndim = len(patch_shape)
    rays = _build_rays(args, ndim)

    train_transform = _build_train_transform(args.pmin, args.pmax, args.augment, args.gauss_std)
    val_transform = InputPercentileNormalize(pmin=args.pmin, pmax=args.pmax)

    train_ds = StarDistH5Dataset(args.h5, split="train", rays=rays, transform=train_transform)
    val_ds = StarDistH5Dataset(args.h5, split="val", rays=rays, transform=val_transform)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=stardist_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=stardist_collate,
    )

    trainer_obj = StarDistTrainer(
        model_name=args.model_name,
        model_dir=args.model_dir,
        rays=rays,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        conv_dims=ndim,
        depth=args.depth,
        num_channels_init=args.channels_init,
    )
    trainer_obj.fit(train_dataloader=train_loader, val_dataloader=val_loader)
    print(f"Trained model + rays sidecar saved under {args.model_dir}/")


# ============================================================== predict

def cmd_predict(args):
    rays = np.load(args.rays)
    seg = StarDistSegmenter.from_checkpoint(
        args.ckpt,
        rays=rays,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
        n_tiles=args.n_tiles,
        conv_dims=rays.shape[1],
        depth=args.depth,
        num_channels_init=args.channels_init,
    )
    image = imread(args.image)
    if image.ndim != rays.shape[1]:
        raise SystemExit(
            f"image ndim={image.ndim} doesn't match rays ndim={rays.shape[1]}"
        )

    result = seg.predict(image)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imwrite(args.out, result.labels.astype(np.uint16))
    print(f"Wrote {result.labels.max()} labels → {args.out}")


# ============================================================== argparse

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # prep
    pp = sub.add_parser("prep", help="generate H5 (raw, label) patches from raw + label dirs")
    pp.add_argument("--raw", required=True, type=Path)
    pp.add_argument("--label", required=True, type=Path)
    pp.add_argument("--out", required=True, type=Path,
                    help="output H5 path")
    pp.add_argument("--patch", required=True, type=int, nargs="+",
                    help="patch shape, 2 ints (Y X) for 2D or 3 (Z Y X) for 3D")
    pp.add_argument("--val-files", type=int, default=1)
    pp.add_argument("--min-fg", type=float, default=0.0,
                    help="drop patches with foreground voxel fraction below this")
    pp.add_argument("--overwrite", action="store_true")
    pp.set_defaults(func=cmd_prep)

    # train
    tp = sub.add_parser("train", help="train StarDist on the prepped H5")
    tp.add_argument("--h5", required=True, type=Path)
    tp.add_argument("--model-dir", required=True, type=Path)
    tp.add_argument("--model-name", required=True, type=str)
    tp.add_argument("--n-rays", type=int, default=96)
    tp.add_argument("--anisotropy", type=float, nargs=3, default=None,
                    help="(Z Y X) voxel spacing — 3D only")
    tp.add_argument("--epochs", type=int, default=100)
    tp.add_argument("--batch-size", type=int, default=4)
    tp.add_argument("--lr", type=float, default=4e-4)
    tp.add_argument("--depth", type=int, default=3)
    tp.add_argument("--channels-init", type=int, default=64)
    tp.add_argument("--pmin", type=float, default=0.1)
    tp.add_argument("--pmax", type=float, default=99.9)
    tp.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader workers — recommended >=2 since target compute happens per sample")
    tp.add_argument("--augment", action="store_true",
                    help="enable random flip + rot90 + gaussian noise")
    tp.add_argument("--gauss-std", type=float, default=0.01,
                    help="stddev for Gaussian noise (only with --augment)")
    tp.set_defaults(func=cmd_train)

    # predict
    rp = sub.add_parser("predict", help="run inference on a single volume")
    rp.add_argument("--ckpt", required=True, type=Path)
    rp.add_argument("--rays", required=True, type=Path,
                    help="rays sidecar saved by `train` (defaults to <model_dir>/<model_name>.rays.npy)")
    rp.add_argument("--image", required=True, type=Path)
    rp.add_argument("--out", required=True, type=Path)
    rp.add_argument("--prob-thresh", type=float, default=0.5)
    rp.add_argument("--nms-thresh", type=float, default=0.4)
    rp.add_argument("--n-tiles", type=int, nargs="+", default=None)
    rp.add_argument("--depth", type=int, default=3)
    rp.add_argument("--channels-init", type=int, default=64)
    rp.set_defaults(func=cmd_predict)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
