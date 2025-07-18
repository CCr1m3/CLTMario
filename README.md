# CLTMario – Imitation-Learning + DQN for Super Mario Bros

A research/teaching playground that trains an agent to play the NES game *Super Mario Bros.* using a hybrid of

1. Behaviour Cloning (BC) from human demonstrations
2. Deep Q-Learning (DQN) with shaped rewards
3. Population-Based Training (PBT) for hyper-parameter optimisation

The codebase is pure Python (PyTorch) and runs on Linux, macOS and Windows.

---

## 1. Quick Start

```bash
# 1) create a Python 3.11 virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate     # Linux / macOS
#  # or on Windows
#  .venv\Scripts\Activate.ps1

# 2) install dependencies
pip install -r requirements_red.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# alternative
python -m pip install -r requirements_red.txt
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# 3) record a short expert run (optional but highly encouraged)
python record_expert_data.py          # GUI pops up – choose a level & play with arrow keys + Shift + Space

# 4) train the population (2 agents by default)
python train_agent.py

# 5) open Tensorboard in a seperate terminal
python summary.py

# 6) watch the best agent
python test_agent.py
```

If you want to skip manual demonstrations you can still train – the agent will start from random play.

---

## 2. Project Layout

```
CLTMario/
├── train_agent.py        ← main entry-point for training (BC + DQN + PBT)
├── record_expert_data.py ← simple pygame GUI to capture expert trajectories
├── test_agent.py         ← load a trained model and run it in the emulator
├── config.yaml           ← central hyper-parameter file
├── utils/
│   ├── actions.py        ← discrete action set (11 buttons combos)
│   ├── env_wrapper.py    ← preprocessing, frame-stacking, skip-frames
│   ├── model.py          ← small CNN with extra feature inputs
│   └── preprocessing.py  ← CV helpers
└── data/                 ← human demonstration *.pt files are saved here
```

---

## 3. Configuration (`config.yaml`)

Key sections you might tweak:

* `env` – frame size, grayscale, frame-skip, number of stacked frames.
* `training` – epochs, learning-rate, batch-size, γ (discount), ε-greedy schedule, λ<sub>BC</sub> and its decay, replay-buffer size, burn-in steps, etc.
* `data.expert_data_path` – where `.pt` demonstration files are stored.

The file is self-documenting; open it and adjust to your liking.

---

## 4. Recording Expert Data

1. Run `python record_expert_data.py`.
2. Select a world–stage combo (e.g. `1-1`).
3. Play – buttons:
   * **Left / Right / Up / Down** – D-pad
   * **Space / S** – A (jump)
   * **Left-Shift / A** – B (run / fire)
4. Press **Esc** to finish.  If you reach the flag or warpzone the trajectory is saved to `data/<world-stage>/expert_<timestamp>.pt`.

Each file stores tensors:
```
{states, prev_actions, actions, delta_x, delta_y, pos_x, pos_y}
```
which are loaded automatically by `train_agent.py`.

---

## 5. Training Details

* **Behaviour Cloning** – Supervised cross-entropy loss on expert actions for the first `bc_epochs`.
* **DQN** – Double-DQN style update with a target network synced every 100 steps.
* **Reward Shaping** – distance travelled, jumps, sub-areas, survival bonus, flag-bonus, etc. (see `calculate_shaped_reward`).
* **Population-Based Training** – 2 agents mutate `lr` and λ<sub>BC</sub>; weight sharing happens every `pbt_exploit_interval` epochs.
* **Replay Buffer** – recency-biased sampling and **no full reset** after exploitation (keeps 60 % newest transitions).
* **Burn-in** – learning is paused for `burn_in_steps` after each exploit to collect on-policy data.
* **Gradient Clipping** – global norm ≤ 10.

Monitor progress in **TensorBoard**:
```bash
tensorboard --logdir runs
```

---

## 6. Testing / Inference

Edit `config.yaml:test` or call directly:
```bash
python test_agent.py --env_name SuperMarioBros-v0 --render
```
The script loads `models/combined_model.pt` (exported after training) and runs until the agent dies or hits `max_steps`.

---

## 7. Troubleshooting

| Problem                                    | Fix                                                               |
|--------------------------------------------|-------------------------------------------------------------------|
| `FrameStack` import error                  | Upgrade Gym `pip install gym==0.26.*` or change import as warned  |
| NES emulator window very slow              | Enable `frame_skip` > 1 or use headless mode (no render)          |
| CUDA OOM                                   | Reduce `batch_size`, image `resize_shape`, or use CPU             |

---

## 8. Citation / Credits

* NES emulator: [nes-py](https://github.com/kimamula/nes-py) & [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
* Base DQN ideas: DeepMind 2015
* PBT: Jaderberg *et al.*, 2017

---
README generated with o3