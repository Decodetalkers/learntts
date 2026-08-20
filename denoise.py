from unet import Unet
from diffusion import Diffusion
from typing import List, Tuple
import torch
import torch.nn as nn
import params
from utils import EMO_FEATURES
from dataset import EmoBatchCollate, EmoBatchCollate, EmoDataset, EmoDB
from torch.utils.tensorboard import SummaryWriter
from tqdm.asyncio import tqdm
from torch.utils.data import DataLoader
from utils import fix_len_compatibility
import numpy as np

save_model = "denoise.pt"


class Denoise(nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        self.unet = Unet(1, 1)
        self.diffusion = Diffusion()

        self.lossfn = torch.nn.MSELoss()

    def denoise(self, x0: torch.Tensor) -> torch.Tensor:
        len = fix_len_compatibility(x0.shape[2])
        y0 = torch.zeros(
            x0.shape[0], x0.shape[1], len, dtype=x0.dtype, device=x0.device
        )
        y0[:, :, : x0.shape[2]] = x0
        y0 = self.forward(y0)
        x0 = y0[:, :, : x0.shape[2]]
        return x0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(dim=1)
        x = self.unet(x)
        x = x.squeeze(dim=1)
        return x

    def diff_train(self, x_0: torch.Tensor) -> torch.Tensor:
        x, _t = self.diffusion.diffuse(x_0)
        x = self.forward(x)
        return x

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        x = self.diff_train(x_0)
        return self.lossfn(x_0, x)


if __name__ == "__main__":
    # what need to take care about is the length of the time. It should always be consisted with n_mels

    print("Initializing logger...")
    logger = SummaryWriter(log_dir=params.log_dir2)

    print("Initializing model...")
    model = Denoise().to(params.device)
    dataset = EmoDataset(EmoDB, n_fft=params.n_fft, n_mels=params.n_mels)
    batch_collate = EmoBatchCollate(
        dataset.min_div, dataset.emo_features, dataset.mels_count
    )

    train, test = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(
        dataset=train,
        shuffle=True,
        batch_size=params.denoise_batch_size,
        collate_fn=batch_collate,
    )
    test_loader = DataLoader(
        dataset=test,
        shuffle=True,
        batch_size=params.denoise_batch_size,
        collate_fn=batch_collate,
    )

    loss_collect: List[float] = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)

    iteration = 0

    for epoch in range(1, params.n_epochs + 1):
        model.train()
        with tqdm(
            train_loader, total=len(train) // params.denoise_batch_size
        ) as progress_bar:
            for batch_idx, (emo, mel) in enumerate(progress_bar):
                model.zero_grad()
                mel = mel.to(params.device)
                loss = model.compute_loss(mel)
                loss.backward()

                loss_collect.append(loss.item())

                optimizer.step()

                logger.add_scalar("training/loss", loss.item(), global_step=iteration)

                iteration += 1

                if batch_idx % 5 == 0:
                    progress_bar.set_description(
                        f"Epoch: {epoch}, iteration: {iteration}, loss: {loss.item():.8f}"
                    )
        model.eval()
        log_msg = "Epoch %d: denoise loss = %.8f\n" % (epoch, np.mean(loss_collect))
        with open(f"{params.log_dir2}/train.log", "a") as f:
            f.write(log_msg)

    torch.save(model.state_dict(), save_model)

    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    plt.figure(figsize=(10, 10))
    iter_count = len(loss_collect)
    x_axis = np.arange(iter_count)
    plt.title("loss iteration")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.ylim(top=3)
    plt.plot(x_axis, loss_collect)
    plt.savefig(Path(params.log_dir2) / "loss_iter.png")
    plt.close()
