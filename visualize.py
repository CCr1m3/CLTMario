from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    import os
    os.system("tensorboard --logdir=runs --port=6006")