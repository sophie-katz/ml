# Copyright (c) 2023 Sophie Katz
#
# This file is part of Sophie's ML Monorepo.
#
# Sophie's ML Monorepo is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# Sophie's ML Monorepo is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Sophie's
# ML Monorepo. If not, see <https://www.gnu.org/licenses/>.

from ml.data.pytorch_name_classification.pytorch import PytorchNameCategorization
from ml.models.name_classification_rnn.model import HIDDEN_SIZE, NameClassificationRNN
import torch as T
import lightning.pytorch as pl
from ml.models.name_classification_rnn.module import NameClassificationRNNModule
from ml.core.repo_paths import get_dir_models


def train() -> NameClassificationRNN:
    dataset = PytorchNameCategorization()

    model = NameClassificationRNN(
        HIDDEN_SIZE,
        dataset.alphabet_count(),
        dataset.culture_name_count(),
    )

    module = NameClassificationRNNModule(dataset, model)

    trainer = pl.Trainer(max_epochs=1)

    trainer.fit(
        module,
        train_dataloaders=T.utils.data.DataLoader(
            dataset,
            sampler=T.utils.data.RandomSampler(dataset, num_samples=10000),
            batch_size=None,
        ),
    )

    return model


if __name__ == "__main__":
    model = train()

    model.save(
        get_dir_models("name_classification_rnn", create=True)
        / "name_classification_rnn.pt"
    )
