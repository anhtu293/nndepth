import argparse

from nndepth.utils.common import add_common_args, instantiate_with_config_file
from nndepth.disparity.train import LitDisparityModel
from nndepth.disparity.data_loaders.tartanair2disparity import TartanairDisparityDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args()

    model = instantiate_with_config_file(args.model_config, "nndepth.disparity.models")
    dataloader = instantiate_with_config_file(args.data_config, "nndepth.disparity.data_loaders")
    trainer = instantiate_with_config_file(args.training_config, "nndepth.disparity.trainers")

    lit = LitDisparityModel(args=args)
    tartan_loader = Tartanair2DisparityModel(args=args)
    if args.expe_name is None:
        expe_name = "disparity_on_tartanair"
    else:
        expe_name = f"disparity-{lit.model_name}-{args.expe_name}"
    lit.run_train(tartan_loader, args, project="disparity", expe_name=expe_name)


if __name__ == "__main__":
    main()
