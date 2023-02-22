import argparse
from train import train
from evaluate import get_test_accuracy


parser = argparse.ArgumentParser(description="Dirichlet Pruning")
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('--mode',
                    choices=["train_original", "train_compressed", "prune", "evaluate_original", "evaluate_compressed"],
                    required=True)

optional.add_argument('--method', choices=["dirichlet", "generalized_dirichlet"], default="None")
optional.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
optional.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
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
                    default='saved_models', type=str)


def main():
    args=parser.parse_args()

    if args.mode == "train_original":
        train(
            model_type=args.mode,
            save_dir=args.save_dir,
            cpu=args.cpu,
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

    elif args.mode == "train_compressed":
        pass
    elif args.mode == "prune":
        pass
    elif args.mode == "evaluate_original":
        pass
    elif args.mode == "evaluate_compressed":
        pass
    else:
        raise Exception("Invalid mode.")


if __name__ == '__main__':
    main()