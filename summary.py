from torch.utils.tensorboard import SummaryWriter
import subprocess

if __name__ == "__main__":
    logdir = "runs"
    print(f"Launching TensorBoard at http://localhost:6006/ (logdir={logdir})")
    subprocess.run(["tensorboard", "--logdir", logdir, "--port", "6006"])
