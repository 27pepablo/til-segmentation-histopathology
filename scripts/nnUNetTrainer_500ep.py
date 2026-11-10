"""nnU-Net v2 trainer limited to 500 epochs (the default is 1000).

All models in the thesis were trained for 500 epochs: preliminary runs showed
the pseudo-Dice had already plateaued, and it halved the training time.

nnU-Net discovers trainers by scanning its own package, so copy this file into
the installed library before training (``scripts/train_nnunet.sh`` does it)::

    NNUNET_DIR="$(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')"
    cp scripts/nnUNetTrainer_500ep.py \
       "$NNUNET_DIR/training/nnUNetTrainer/variants/training_length/"
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_500ep(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),  # noqa: B008
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
