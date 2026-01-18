import os
import argparse
from datetime import datetime

from matformer.train_props import train_prop_model


def run_task(prop: str, args):
    out_dir = os.path.join(args.output_root, f"{prop}_{args.link}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[START] {prop} | out_dir={out_dir}")
    train_prop_model(
        dataset="edos_pdos",
        prop=prop,
        name="matformer",
        pyg_input=True,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_lattice=not args.no_lattice,
        use_angle=args.use_angle,
        output_dir=out_dir,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        num_workers=args.num_workers,
        save_dataloader=args.save_dataloader,
        link=args.link,
        random_seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        write_checkpoint=False,
        weighted_loss=args.weighted_loss,
        per_bin_standardize=args.per_bin_standardize,
        fermi_weight_sigma=args.fermi_sigma,
        fermi_bin_idx=args.fermi_bin_idx,
    )
    print(f"[DONE] {prop} | outputs saved under: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="One-click runner for EDOS/pDOS training with Matformer"
    )
    parser.add_argument("--task", choices=["edos_up", "pdos_elast", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--save_dataloader", action="store_true")
    parser.add_argument("--output_root", default=os.path.join(".", "runs"))
    parser.add_argument("--link", choices=["log", "identity", "logit"], default="log")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no_lattice", action="store_true", help="Disable lattice features")
    parser.add_argument("--use_angle", action="store_true", help="Enable angle features if supported")
    parser.add_argument("--train_ratio", type=float, default=None, help="Subset ratio for train split (e.g., 0.01)")
    parser.add_argument("--val_ratio", type=float, default=None, help="Subset ratio for val split (e.g., 0.005)")
    parser.add_argument("--test_ratio", type=float, default=None, help="Subset ratio for test split (e.g., 0.005)")
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--per_bin_standardize", action="store_true")
    parser.add_argument("--fermi_sigma", type=float, default=None)
    parser.add_argument("--fermi_bin_idx", type=int, default=None)

    args = parser.parse_args()
    print(f"[CONFIG] {vars(args)}")

    t0 = datetime.now()
    if args.task in ("edos_up", "both"):
        run_task("edos_up", args)
    if args.task in ("pdos_elast", "both"):
        run_task("pdos_elast", args)
    t1 = datetime.now()
    print(f"[TOTAL TIME] {(t1 - t0)}")


if __name__ == "__main__":
    main()