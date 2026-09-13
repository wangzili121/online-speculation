"""Full-vocabulary prefix-overlap continuation in the shared paired benchmark."""

from blockspec.commands.continue_training import argument_parser, run_experiment
from blockspec_ablation.prefix_objective import PrefixConfig, PrefixLearner, feedback_factory


def main():
    parser = argument_parser(loss_choices=("prefix_overlap",))
    parser.description = __doc__
    parser.set_defaults(temperature=1., top_k=0, top_p=1.)
    args = parser.parse_args()
    if args.temperature <= 0 or args.top_k or args.top_p != 1:
        parser.error("prefix-overlap uses a positive-temperature full-vocabulary proposal")
    config = PrefixConfig(args.last_layers, args.stride, args.replay_blocks, args.learning_rate,
                          temperature=args.temperature)
    run_experiment(args, config=config, learner_factory=PrefixLearner, feedback_factory=feedback_factory)


if __name__ == "__main__":
    main()
