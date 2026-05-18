import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def list_images(p: Path):
    return sorted([x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS])

def stem(p: Path): return p.stem

def load_openclip(device, model_name):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model = model.to(device).eval()
    return model, preprocess

@torch.no_grad()
def encode_batch(model, x):
    z = model.encode_image(x)
    z = F.normalize(z, dim=-1)
    return z

def pil_load_rgb(p: Path):
    return Image.open(p).convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)  # contains real/ fake/
    ap.add_argument("--test_dir", type=str, required=True)
    ap.add_argument("--out_npz", type=str, default="clip_feats.npz")
    ap.add_argument("--clip_model", type=str, default="ViT-L-14-336")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    train_dir = Path(args.train_dir)
    test_dir  = Path(args.test_dir)

    real_paths = list_images(train_dir / "real")
    fake_paths = list_images(train_dir / "fake")
    test_paths = list_images(test_dir)

    if not real_paths or not fake_paths:
        raise FileNotFoundError("train_dir 需要有 real/ 與 fake/ 並含圖片")
    if not test_paths:
        raise FileNotFoundError("test_dir 找不到圖片")

    print(f"[INFO] real={len(real_paths)} fake={len(fake_paths)} test={len(test_paths)} device={device}")
    model, preprocess = load_openclip(device, args.clip_model)

    # ---- prepare lists ----
    train_paths = real_paths + fake_paths
    y = np.array([0]*len(real_paths) + [1]*len(fake_paths), dtype=np.int64)  # fake=1
    train_names = np.array([stem(p) for p in train_paths])
    test_names  = np.array([stem(p) for p in test_paths])

    def run_encode(paths, tta=False):
        feats = []
        feats_tta = [] if tta else None

        bs = args.batch_size
        for i in range(0, len(paths), bs):
            batch_paths = paths[i:i+bs]
            imgs = [preprocess(pil_load_rgb(p)) for p in batch_paths]
            x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
            z = encode_batch(model, x).cpu().numpy()
            feats.append(z)

            if tta:
                imgs2 = [preprocess(pil_load_rgb(p).transpose(Image.FLIP_LEFT_RIGHT)) for p in batch_paths]
                x2 = torch.stack(imgs2, dim=0).to(device, non_blocking=True)
                z2 = encode_batch(model, x2).cpu().numpy()
                feats_tta.append(z2)

            if (i//bs) % 20 == 0:
                print(f"  encoded {min(i+bs, len(paths))}/{len(paths)}")

        feats = np.concatenate(feats, axis=0).astype(np.float32)
        if tta:
            feats_tta = np.concatenate(feats_tta, axis=0).astype(np.float32)
            return feats, feats_tta
        return feats

    print("[INFO] Encoding train...")
    X_train = run_encode(train_paths, tta=False)

    print("[INFO] Encoding test (with TTA)...")
    X_test, X_test_flip = run_encode(test_paths, tta=True)

    np.savez(
        args.out_npz,
        train_names=train_names,
        y=y,
        X_train=X_train,
        test_names=test_names,
        X_test=X_test,
        X_test_flip=X_test_flip,
        clip_model=args.clip_model
    )
    print(f"[OK] Saved cache: {args.out_npz}")

if __name__ == "__main__":
    main()
