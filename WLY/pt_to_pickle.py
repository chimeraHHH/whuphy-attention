import argparse
import os
import pickle
import torch


def convert(input_dir, output_path, limit=None):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    pt_files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
    pt_files.sort()
    if limit is not None:
        pt_files = pt_files[:limit]

    samples = []
    for fname in pt_files:
        fpath = os.path.join(input_dir, fname)
        data = torch.load(fpath, map_location="cpu", weights_only=False)

        numbers = data.x.view(-1).to(torch.long).tolist()
        positions = data.pos.to(torch.float32).cpu().numpy()

        lattice = getattr(data, "lattice", None)
        if lattice is None:
            raise ValueError(f"Missing lattice in {fpath}")
        cell = lattice.squeeze(0).to(torch.float32).cpu().numpy()

        y = data.y.view(-1)
        if y.numel() < 1:
            raise ValueError(f"Missing y in {fpath}")
        target = float(y[0].item())

        uid = getattr(data, "mp_id", None)
        if uid is None:
            uid = os.path.splitext(fname)[0]

        samples.append(
            {
                "id": uid,
                "numbers": numbers,
                "positions": positions,
                "cell": cell,
                "pbc": True,
                "target": target,
            }
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)

    print(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=os.path.join(base_dir, "dataset_imp2d"))
    parser.add_argument("--output", default=os.path.join(base_dir, "cleaned_dataset.pkl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    convert(args.input_dir, args.output, args.limit)
