from torch.utils.tensorboard import SummaryWriter
from tqdm.asyncio import tqdm
from torch.utils.data import DataLoader
from typing import List
import torch
from unet import Unet
import params
from utils import EMO_EFATURES
from dataset import EmoBatchCollate, EmoBatchCollate, EmoDataset, EmoDB


save_model = "emo_classify_01.pt"


# this module will train a random shape data to 4 grade
# flat the n * m data and add embedding
class EmoClassify(torch.nn.Module):
    convs: List[torch.nn.Module]

    def __init__(
        self,
        n_mels: int,
        out_features: int,
        hidden_dim: int = 1024,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super(EmoClassify, self).__init__()

        self.softmax = torch.nn.Softmax(dim=-1)
        self.unet = Unet(1000, 128, in_channels=1, out_channels=1)

        self.lstm = torch.nn.LSTM(
            n_mels,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.linear = torch.nn.Linear(hidden_dim, out_features)

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        x_0 = x_0.unsqueeze(dim=1)
        x_0 = self.unet(x_0)
        x_0 = x_0.squeeze()
        x_0 = x_0.transpose(-1, -2)
        x_0, (_hidden, _cell) = self.lstm(x_0)
        x_0 = self.linear(x_0)
        x_0 = x_0.mean(dim=-2)
        x_0 = self.softmax(x_0)

        return x_0


@torch.no_grad()
def compute_accuracy(model: EmoClassify, data_loader: DataLoader) -> float:
    correct_pred, num_examples = torch.tensor(0, dtype=torch.int64).to(params.device), 0
    for emo, mel in data_loader:
        predict = model(mel.to(params.device))
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
    model = EmoClassify(params.n_mels, EMO_EFATURES).to(params.device)
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

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)

    iteration = 0
    lossfn = torch.nn.CrossEntropyLoss()
    for epoch in range(1, params.n_epochs + 1):
        model.train()
        with tqdm(train_loader, total=len(train) // params.batch_size) as progress_bar:
            for batch_idx, (emo, mel) in enumerate(progress_bar):
                model.zero_grad()
                predict = model(mel.to(params.device))
                loss = lossfn(predict.cpu(), emo.float())
                loss.backward()

                loss_collect.append(loss.item())

                optimizer.step()

                logger.add_scalar("training/loss", loss.item(), global_step=iteration)

                iteration += 1

                if batch_idx % 5 == 0:
                    progress_bar.set_description(
                        f"Epoch: {epoch}, iteration: {iteration}, loss: {loss.item()}"
                    )
        model.eval()
        with torch.set_grad_enabled(False):
            accuracy = compute_accuracy(model, test_loader)
            print(f"Epoch: {epoch}/{params.n_epochs} training accuracy: {accuracy}%")

    torch.save(model.state_dict(), save_model)
