import torch
import pandas as pd

_db = csv = pd.read_csv("./EmoSpeech-0020/0020.txt", sep="\t", header=None)

_natural = "Neutral"
_surprise = "Surprise"
_angry = "Angry"
_happy = "Happy"
_sad = "Sad"


class EmoDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, n_mels: int):
        self.emodb = data
        self.n_mels = n_mels

    def __len__(self) -> int:
        return self.emodb.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        # wav_file = self.emodb[0][index]
        emo: str = self.emodb[2][index]
        emo_data: torch.Tensor
        match emo:
            case "Surprise":
                emo_data = torch.tensor([0, 1, 0, 0, 0])
            case "Angry":
                emo_data = torch.tensor([0, 0, 1, 0, 0])
            case "Happy":
                emo_data = torch.tensor([0, 0, 0, 1, 0])
            case "Sad":
                emo_data = torch.tensor([0, 0, 0, 0, 1])
            case _:
                emo_data = torch.tensor([1, 0, 0, 0, 0])
        return emo_data


if __name__ == "__main__":
    dataset = EmoDataset(_db, 80)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=10)
    for emo_data in loader:
        print(emo_data)
        break
