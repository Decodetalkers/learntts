from torch.utils.tensorboard import SummaryWriter
from tqdm.asyncio import tqdm
from torch.utils.data import DataLoader
from typing import List, Tuple
import torch
from unet import Unet
import params
from utils import EMO_FEATURES
from dataset import EmoBatchCollate, EmoBatchCollate, EmoDataset, EmoDB
from classification import EmoClassification

save_model = "emo_classify_01.pt"


# this module will train a random shape data to 4 grade
# flat the n * m data and add embedding
class EmoClassify(torch.nn.Module):
    convs: List[torch.nn.Module]

    def __init__(
        self,
        n_mels: int,
        out_features: int,
        tau: float = 0.01,
    ):
        super(EmoClassify, self).__init__()

        self.tau = tau

        self.classify = EmoClassification(in_channels=n_mels, out_channels=out_features)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_0, t = self.classify(x_0)
        return x_0, t

    def train_label(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        this part make the biggest label stronger, in order to predict the right label
        Only used in training process
        """
        x_t, t = self.forward(x_0)
        t_w = 1.0 - t
        t_w = - torch.log(t_w ** 6) + 1.0
        x_t = x_t * t_w[:, None]
        loss = self.softmax(x_t / self.tau)
        return loss


@torch.no_grad()
def compute_accuracy(model: EmoClassify, data_loader: DataLoader) -> float:
    correct_pred, num_examples = torch.tensor(0, dtype=torch.int64).to(params.device), 0
    for emo, mel in data_loader:
        predict, _ = model(mel.to(params.device))
        predict_labels = predict.argmax(dim=1)
        emo_labels = emo.to(params.device).argmax(dim=1)
        sum = (predict_labels == emo_labels).sum()
        correct_pred += sum
        num_examples += params.batch_size
    return correct_pred.float().item() / num_examples * 100


if __name__ == "__main__":
    # what need to take care about is the length of the time. It should always be consisted with n_mels

    print("Initializing logger...")
    logger = SummaryWriter(log_dir=params.log_dir)

    print("Initializing model...")
    model = EmoClassify(params.n_mels, EMO_FEATURES).to(params.device)
    dataset = EmoDataset(EmoDB, n_fft=params.n_fft, n_mels=params.n_mels)
    batch_collate = EmoBatchCollate(
        dataset.min_div, dataset.emo_features, dataset.mels_count
    )

    train, test = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(
        dataset=train,
        shuffle=True,
        batch_size=params.batch_size,
        collate_fn=batch_collate,
    )
    test_loader = DataLoader(
        dataset=test,
        shuffle=True,
        batch_size=params.batch_size,
        collate_fn=batch_collate,
    )

    loss_collect: List[float] = []

    accuracy_collect: Tuple[List[float], List[float]] = ([], [])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)

    iteration = 0
    lossfn = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    for epoch in range(1, params.n_epochs + 1):
        model.train()
        with tqdm(train_loader, total=len(train) // params.batch_size) as progress_bar:
            for batch_idx, (emo, mel) in enumerate(progress_bar):
                model.zero_grad()
                predict = model.train_label(mel.to(params.device))
                loss = lossfn(predict, emo.argmax(dim=1).to(params.device))
                loss.backward()

                loss_collect.append(loss.item())

                optimizer.step()

                logger.add_scalar("training/loss", loss.item(), global_step=iteration)

                iteration += 1

                if batch_idx % 5 == 0:
                    progress_bar.set_description(
                        f"Epoch: {epoch}, iteration: {iteration}, loss: {loss.item():.2f}"
                    )
        model.eval()
        with torch.set_grad_enabled(False):
            accuracy_train = compute_accuracy(model, train_loader)
            accuracy_collect[0].append(accuracy_train)
            print(
                f"Epoch: {epoch}/{params.n_epochs} training accuracy: {accuracy_train:.2f}%"
            )
            accuracy_test = compute_accuracy(model, test_loader)
            print(
                f"Epoch: {epoch}/{params.n_epochs} testing accuracy: {accuracy_test:.2f}%"
            )
            accuracy_collect[1].append(accuracy_test)

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
    plt.savefig(Path(params.log_dir) / "loss_iter.png")
    plt.close()

    plt.figure(figsize=(10, 10))
    accuracy_train_l, accuracy_test_l = accuracy_collect
    epoch_count = len(accuracy_collect[0])
    x_axis = np.arange(epoch_count)
    plt.title("Accuracy Figure")
    plt.xlabel("epoch")
    plt.ylabel("accuracy %")
    plt.ylim(top=100)
    plt.plot(x_axis, accuracy_test_l, color="y", label="Test")
    plt.plot(x_axis, accuracy_train_l, color="b", label="Train")
    plt.legend(loc="upper right")
    plt.savefig(Path(params.log_dir) / "accuracy.png")
    plt.close()
