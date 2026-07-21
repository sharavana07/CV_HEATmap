from cnn.config import (
    MODEL_NAME,
    MODEL_CNN,
    MODEL_CNN_SE,
    MODEL_DAFNET,
    DAFNET_CONFIG,
)

from cnn.model import OrderBookCNN as BaseOrderBookCNN
from cnn.model_se import OrderBookCNN as SEOrderBookCNN
from cnn.model_Star1 import build_model


def get_model(model_name=None):
    model_name = model_name or MODEL_NAME

    if model_name == MODEL_CNN:
        return BaseOrderBookCNN()

    if model_name == MODEL_CNN_SE:
        return SEOrderBookCNN()

    if model_name == MODEL_DAFNET:
        return build_model(DAFNET_CONFIG)

    raise ValueError(f"Unknown model: {model_name}")