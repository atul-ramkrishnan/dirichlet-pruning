import argparse
from train import train, train_importance_switches
from evaluate import get_test_accuracy
from prune import prune_and_retrain
import torch


parser = argparse.ArgumentParser(description="Dirichlet Pruning")
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('--mode',
                    choices=["train_original", "train_importance_switches", "prune_and_retrain", "evaluate_original", "evaluate_compressed"],
                    required=True)

optional.add_argument('--method', choices=["dirichlet", "generalized_dirichlet"], default="None")
optional.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
optional.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
optional.add_argument('--original', default='', type=str, metavar='PATH',
                      help='path to original model checkpoint (default: none)')
optional.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
optional.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
optional.add_argument('--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
optional.add_argument('--lr', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
optional.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
optional.add_argument('--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
optional.add_argument('--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
optional.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
optional.add_argument('--save-dir',
                    help='The directory used to save the trained models',
                    default='saved', type=str)
optional.add_argument("--switch_samps", default=150, type=int)
optional.add_argument("--start-layer", default="conv1", type=str, metavar='L', 
                      help='resume training importance switches from layer L')
optional.add_argument("--create-bkp", action='store_true', default=True,
                      help='Creates backup of checkpoints during imp switch training (default: True)')


def main():
    args=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Running on {device}...")
    if args.mode == "train_original":
        print("Training the original (uncompressed) network...")
        train(
            model_type=args.mode,
            save_dir=args.save_dir,
            device=device,
            resume=args.resume,
            eval=args.evaluate,
            batch_size=args.batch_size,
            workers=args.workers,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            start_epoch=args.start_epoch,
            epochs=args.epochs,
            print_freq=args.print_freq
            )

    elif args.mode == "train_importance_switches":
        print(f"Training the importance switches using the \"{args.method}\" method...")

        train_importance_switches(
                                method=args.method,
                                switch_samps=args.switch_samps,
                                device=device,
                                original=args.original,
                                resume=args.resume,
                                batch_size=args.batch_size,
                                workers=args.workers,
                                lr=args.lr,
                                start_layer=args.start_layer,
                                start_epoch=args.start_epoch,
                                epochs=args.epochs,
                                print_freq=args.print_freq,
                                save_dir=args.save_dir,
                                create_bkp=args.create_bkp
                                )
    elif args.mode == "prune_and_retrain":
        print("Pruning and retraining...")

        prune_and_retrain(
                        switch_save_path=args.switch_save_path,
                        thresholds=args.thresholds,
                        model_save_dir=args.save_dir,
                        device=device,
                        resume=args.resume,
                        eval=args.evaluate,
                        batch_size=args.batch_size,
                        workers=args.workers,
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        start_epoch=args.start_epoch,
                        epochs=args.epochs,
                        print_freq=args.print_freq
                        )
    elif args.mode == "evaluate_original":
        pass
    elif args.mode == "evaluate_compressed":
        pass
    else:
        raise Exception("Invalid mode.")


if __name__ == '__main__':
    main()