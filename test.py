from quad_net import QuadNet
from torch.utils.data import DataLoader
import torch
from quad_dataset import QuadDataset
from sklearn.metrics import r2_score


def test():
    BATCH_SIZE = 2000
    dataset = QuadDataset(is_train=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = QuadNet()
    model.load_state_dict(torch.load("models/saha.h5"))
    model.eval()
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = None
    print(f"Test started ...")
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            y_pred = y_pred.reshape(-1)
            loss = criterion(y_pred, y_true)
            print(f"Loss {loss}")
            print(f"R2 {r2_score(y_true, y_pred)}")

if __name__ == "__main__":
    test()
