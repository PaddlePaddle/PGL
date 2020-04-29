from models.base import BaseGNNModel
from models.ernie import ErnieModel
from models.erniesage_v1 import ErnieSageModelV1
from models.erniesage_v2 import ErnieSageModelV2
from models.erniesage_v3 import ErnieSageModelV3

class Model(object):
    @classmethod
    def factory(cls, config):
        name = config.model_type
        if name == "BaseGNNModel":
            return BaseGNNModel(config)
        if name == "ErnieModel":
            return ErnieModel(config)
        if name == "ErnieSageModelV1":
            return ErnieSageModelV1(config)
        if name == "ErnieSageModelV2":
            return ErnieSageModelV2(config)
        if name == "ErnieSageModelV3":
            return ErnieSageModelV3(config)
        else:
            raise ValueError


