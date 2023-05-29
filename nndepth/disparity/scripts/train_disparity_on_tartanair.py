import argparse

import alonet

from nndepth.disparity.train import LitDisparityModel
from nndepth.disparity.data_modules.tartanair2disparity import Tartanair2DisparityModel


def main():
    parser = argparse.ArgumentParser()
    parser = alonet.common.pl_helpers.add_argparse_args(parser, add_pl_args=True, mode="training")
    parser = Tartanair2DisparityModel.add_argparse_args(parser)
    parser = LitDisparityModel.add_argparse_args(parser)
    args = parser.parse_args()

    lit = LitDisparityModel(args)
    tartan_loader = Tartanair2DisparityModel(args)
    if args.expe_name is None:
        expe_name = "disparity_on_tartanair"
    else:
        expe_name = f"disparity-{args.model_name}-{args.expe_name}"
    lit.run_train(tartan_loader, args, project="disparity", expe_name=expe_name)


if __name__ == "__main__":
    main()
