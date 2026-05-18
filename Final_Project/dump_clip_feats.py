import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}

def list_images(p: Path):
    return sorted([x for x in p.rglob("*") if x.suffix.lower() in IMG_EXTS])

def stem_name(p: Path) -> str:
    return p.stem

def load_openclip(device, model_name):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model = model.to(device).eval()
    return model, preprocess

@torch.no_grad()
def encode(model, preprocess, paths, device, batch_size=64):
    feats = []
    names = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = []
        for p in batch:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        x = torch.stack(imgs, 0).to(device)
        z = model.encode_image(x)
        z = F.normalize(z, dim=-1)
        feats.append(z.cpu().numpy())
        names += [stem_name(p) for p in batch]
    return np.concatenate(feats, 0).astype(np.float32), np.array(names)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--clip_model", required=True)
    ap.add_argument("--out_prefix", required=True, help="輸出前綴，例如 feats_L14_336")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    train_dir = Path(args.train_dir)
    test_dir  = Path(args.test_dir)

    real_paths = list_images(train_dir/"real")
    fake_paths = list_images(train_dir/"fake")
    test_paths = list_images(test_dir)

    X_paths = real_paths + fake_paths
    y = np.array([0]*len(real_paths) + [1]*len(fake_paths), dtype=np.int64)

    model, preprocess = load_openclip(device, args.clip_model)

    X_tr, name_tr = encode(model, preprocess, X_paths, device, args.batch_size)
    X_te, name_te = encode(model, preprocess, test_paths, device, args.batch_size)

    np.save(f"{args.out_prefix}_X_train.npy", X_tr)
    np.save(f"{args.out_prefix}_y_train.npy", y)
    np.save(f"{args.out_prefix}_name_train.npy", name_tr)
    np.save(f"{args.out_prefix}_X_test.npy", X_te)
    np.save(f"{args.out_prefix}_name_test.npy", name_te)

    print("[OK] saved:",
          f"{args.out_prefix}_X_train.npy",
          f"{args.out_prefix}_X_test.npy")

if __name__ == "__main__":
    main()

