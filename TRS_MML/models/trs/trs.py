from .unit import UniT
from mmf.common.registry import registry


@registry.register_model("trs")
class TRS(UniT):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "TRS_MML/configs/models/trs/defaults.yaml"
