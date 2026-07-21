import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cnn import config as cfg
from cnn.train import build_arg_parser


def test_build_arg_parser_defaults_to_config_model_name():
    parser = build_arg_parser()
    args = parser.parse_args([])

    assert args.model == cfg.MODEL_NAME
