import argparse
from experiments.dqn import train as dqn_train
from experiments.dqn import evaluate as dqn_evaluate
from experiments.dqn import train_with_memory as dqn_train_with_memory
from experiments.dqn import evaluate_with_memory as dqn_evaluate_with_memory
from experiments.actor_critic import train as ac_train
# from experiments.actor_critic import evaluate as ac_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['dqn', 'dqn_with_memory', 'actor_critic'], default='dqn_with_memory', help='Algorithm to run')
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train', help='Mode to run')
    parser.add_argument('--config_path', type=str, default='configs/lightning_3.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    if args.algo == 'dqn':
        if args.mode == 'train':
            dqn_train.main(config_path=args.config_path)
        else:
            dqn_evaluate.main(config_path=args.config_path)
    elif args.algo == 'dqn_with_memory':
        if args.mode == 'train':
            dqn_train_with_memory.main(config_path=args.config_path)
        else:
            dqn_evaluate_with_memory.main(config_path=args.config_path)
    else:
        if args.mode == 'train':
            ac_train.main(config_path=args.config_path)
        # else:
            # ac_evaluate.main(config_path=args.config_path)


if __name__ == "__main__":
    main()
