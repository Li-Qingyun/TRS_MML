"""
Date: 2022.02.25
Author: Qingyun Li
"""


class FlopsObtainer:
    """
    Three styles for analysing flops are supported:
    fvcore: refer to https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md
    thop: refer to https://github.com/Lyken17/pytorch-OpCounter
    ptflops: refer to https://github.com/sovrasov/flops-counter.pytorch

    The style 'fvcore' is default and recommended.
    """

    def __init__(self, style='fvcore'):
        self._style = style
        if self.style == 'fvcore':
            from fvcore.nn import FlopCountAnalysis, flop_count_table
            self.toolkit_cls = FlopCountAnalysis
            self.print_func = flop_count_table
            self.get_flops_func = self.get_flops_with_fvcore
        elif self.style == 'thop':
            from thop import profile
            self.toolkit_cls = profile
            self.get_flops_func = self.get_flops_with_thop
        elif self.style == 'ptflops':
            from ptflops import get_model_complexity_info
            self.toolkit_func = get_model_complexity_info
            self.get_flops_func = self.get_flops_with_ptflops
        else:
            raise NotImplementedError(
                'The style {} is not supported.'.format(self.style)
            )

    def __call__(self, *args, **kwargs):
        return self.get_flops_func(*args, **kwargs)

    @property
    def style(self):
        return self._style

    def get_flops_with_fvcore(self, model, input):
        flops = self.toolkit_cls(model, input)
        flops.total()
        print(self.print_func(flops, max_depth=2, activations=None, show_param_shapes=False))
