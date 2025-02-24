import argparse
from training.train_agents import main as train_main
from evaluation.evaluate_agents import main as eval_main
from evaluation.plot_results import main as plot_main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "plot"], required=True, help="Choose mode: train, evaluate, plot")
    args = parser.parse_args()

    if args.mode == "train":
        train_main()
    elif args.mode == "evaluate":
        eval_main()
    elif args.mode == "plot":
        plot_main()

if __name__ == "__main__":
    main()