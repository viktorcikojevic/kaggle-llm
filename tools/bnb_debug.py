from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import bitsandbytes as bnb
from accelerate import Accelerator
from tqdm import tqdm
from peft import prepare_model_for_int8_training


DATA_DIM = 4096


class Classifier(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.linear0 = nn.Linear(DATA_DIM, 1)

    def forward(self, x):
        return self.linear0(x)


def generate_data(num: int):
    x = torch.rand((num, DATA_DIM)) * 2 - 1
    y = torch.where(
        (x[:, 0] > 0),
        torch.ones((num, )),
        torch.zeros((num, )),
    )
    return x, y.unsqueeze(1)


def main():
    use_8bits = True
    device = "cuda"
    train_x, train_y = generate_data(1000)
    val_x, val_y = generate_data(100)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    model = Classifier().to(device)
    lr = 0.2
    if use_8bits:
        optim = bnb.optim.Adam8bit(model.parameters(), lr=lr)
        model.linear0 = bnb.nn.Linear8bitLt(
            model.linear0.weight.data.shape[1],
            model.linear0.weight.data.shape[0],
            has_fp16_weights=False,
            threshold=6.0,
            device=device,
            bias=True,
        )
        model.linear0.weight.cuda(device)
        for param in model.parameters():
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)
        # model.linear0.to(device)
        # model.linear0.init_8bit_state()
        # model = prepare_model_for_int8_training(model)
        # model.linear0.register_forward_hook(lambda _m, _i, _o: _o.requires_grad_(True))
        # train_x = train_x.to(torch.float16)
        # val_x = val_x.to(torch.float16)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=lr)

    accelerator = Accelerator()
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=10)
    model, optim, train_loader = accelerator.prepare(model, optim, train_loader)

    epochs = 1000
    for i in range(epochs):
        for x, y in tqdm(train_loader, total=len(train_loader), disable=True):
            optim.zero_grad()
            model = model.train()
            train_pred = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred, y)
            # loss.backward()
            accelerator.backward(loss)
            optim.step()
        if i % 100 != 0:
            continue
        with torch.no_grad():
            model = model.eval()
            print(f"---")
            # print(f"model.linear0.weight.data\n{model.linear0.weight.data}")
            val_pred = model(val_x) > 0
            acc = (val_pred == val_y).float().mean()
            print(f"epoch[{i}]: acc = {acc}")
            print(f"---")


if __name__ == "__main__":
    main()
