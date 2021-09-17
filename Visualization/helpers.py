import logging
import torch.utils.model_zoo as model_zoo

_logger = logging.getLogger(__name__)


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning(
            "Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(
        cfg['url'], progress=False, map_location='cpu')

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)
