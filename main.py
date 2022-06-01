from argparse import ArgumentParser

from data import get_dataset
from model import get_model
from trainer import build_trainer


def main(args):
    # Load model
    model = get_model(args)

    # Load dataset
    data = get_dataset(model)

    # Get Trainer
    trainer = build_trainer(args)

    if "train" in args.mode:
        trainer.fit(model, data)
    elif "test" in args.mode:
        trainer.test(model, data, verbose=True)
    else:
        print(f"Unrecognized mode: {args.mode}")


if __name__ == "__main__":
    parser = ArgumentParser()
    # Model
    parser.add_argument("--model_name", type=str, default="retrosum")
    parser.add_argument("--from_checkpoint", type=str, default="")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--n_neighbors", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=5)

    # Data
    parser.add_argument("--data_name", type=str, default="arxiv")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)

    # Trainer
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--val_check_interval", type=float, default=0.5)
    parser.add_argument("--monitor", type=str, default="val_loss")
    parser.add_argument("--results_filename", type=str, default="results.json")
    parser.add_argument("--fast_dev_run", action="store_true")

    args = parser.parse_args()
    main(args)
