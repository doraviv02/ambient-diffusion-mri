"""Recreate architecture-critical options from a published ``training_options.json``.

The MVP must *not* build the fine-tuning / inference network from the current
``train.py`` defaults (the published R=4 model uses ``channel_mult=[1,1,1,1]``,
``resample_filter=[1,1]`` etc., which differ from those defaults).  This helper
reads the checkpoint's ``training_options.json`` and returns the exact
``network_kwargs`` plus the interface kwargs needed to reconstruct the network,
so a strict (require_all) state-dict load will succeed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import dnnlib


def load_training_options(path: str) -> Dict[str, Any]:
    with dnnlib.util.open_url(path) as f:
        return json.load(f)


def interface_kwargs_from_options(opts: Dict[str, Any], img_channels_image: int = 2) -> Dict[str, Any]:
    """Return ``img_resolution``, ``label_dim`` and ``img_channels`` for the net.

    ``img_channels_image`` is the number of *image* channels (2 for complex).
    Ambient / ambient_mv models take twice that many input channels (image +
    mask), matching the original ``2*num_channels`` convention.
    """
    ds = opts["dataset_kwargs"]
    loss_name = opts.get("loss_kwargs", {}).get("class_name", "")
    is_ambient = "Ambient" in loss_name
    label_dim = 0  # MVP is unconditional
    # Honor the training-time overrides when present. This is essential: SongUNet
    # layer names embed the spatial resolution (e.g. "dec.96x96_block0"), so a
    # fine-tuned model trained with img_resolution_override=384 on 256-px data
    # must be reconstructed at 384 or its state_dict keys will not match.
    img_resolution = opts.get("img_resolution_override") or ds["resolution"]
    img_channels = opts.get("img_channels_override") or (
        img_channels_image * 2 if is_ambient else img_channels_image)
    return dict(
        img_resolution=int(img_resolution),
        label_dim=label_dim,
        img_channels=int(img_channels),
    )


def network_kwargs_from_options(opts: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the checkpoint's ``network_kwargs`` (architecture-critical)."""
    nk = dict(opts["network_kwargs"])
    return nk


def build_network_from_options(opts: Dict[str, Any], img_channels_image: int = 2,
                               dropout: float = None, use_fp16: bool = None):
    """Construct the network exactly as described by ``training_options.json``.

    Only ``dropout`` / ``use_fp16`` may be overridden (they do not change the
    set/shape of trainable parameters).  Everything else is taken verbatim.
    """
    network_kwargs = network_kwargs_from_options(opts)
    interface_kwargs = interface_kwargs_from_options(opts, img_channels_image)
    if dropout is not None:
        network_kwargs["dropout"] = dropout
    if use_fp16 is not None:
        network_kwargs["use_fp16"] = use_fp16
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    return net, network_kwargs, interface_kwargs


def summarize_architecture(opts: Dict[str, Any]) -> str:
    nk = opts["network_kwargs"]
    ds = opts["dataset_kwargs"]
    loss_name = opts.get("loss_kwargs", {}).get("class_name", "")
    is_ambient = "Ambient" in loss_name
    ik = interface_kwargs_from_options(opts)
    lines = [
        "Architecture summary (from training_options.json):",
        f"  precond/class      : {nk.get('class_name')}",
        f"  loss               : {loss_name}",
        f"  model_type         : {nk.get('model_type')}",
        f"  embedding_type     : {nk.get('embedding_type')}",
        f"  encoder/decoder    : {nk.get('encoder_type')}/{nk.get('decoder_type')}",
        f"  model_channels     : {nk.get('model_channels')}",
        f"  channel_mult       : {nk.get('channel_mult')}",
        f"  channel_mult_noise : {nk.get('channel_mult_noise')}",
        f"  resample_filter    : {nk.get('resample_filter')}",
        f"  gated              : {nk.get('gated')}",
        f"  dropout            : {nk.get('dropout')}",
        f"  use_fp16           : {nk.get('use_fp16')}",
        f"  resolution         : {ds.get('resolution')}",
        f"  corruption_prob(R) : {ds.get('corruption_probability')}",
        f"  delta_probability  : {ds.get('delta_probability')}",
        f"  ambient input      : {is_ambient} -> img_channels={ik['img_channels']} (2 image + 2 mask)",
    ]
    return "\n".join(lines)
