import os
import yaml
import torch
import pygame
import tkinter as tk
import numpy as np
import cv2
import datetime
from collections import deque
from tkinter import simpledialog, ttk
from utils.env_wrapper import make_env_human
from utils.actions import CUSTOM_MOVEMENT

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_action_from_keys(keys):
    btn_left = keys[pygame.K_LEFT]
    btn_right = keys[pygame.K_RIGHT]
    btn_up = keys[pygame.K_UP]
    btn_down = keys[pygame.K_DOWN]
    btn_a = keys[pygame.K_SPACE] or keys[pygame.K_s]
    btn_b = keys[pygame.K_LSHIFT] or keys[pygame.K_a]

    pressed = []
    if btn_left: pressed.append('left')
    if btn_right: pressed.append('right')
    if btn_up: pressed.append('up')
    if btn_down: pressed.append('down')
    if btn_a: pressed.append('A')
    if btn_b: pressed.append('B')

    if pressed in CUSTOM_MOVEMENT:
        return CUSTOM_MOVEMENT.index(pressed)
    else:
        return 0 # NOOP

def select_stage():
    valid_stages = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]

    def get_selection():
        selection = []
        def on_submit():
            stage = combo.get()
            if stage in valid_stages:
                selection.append(stage)
                root.destroy()
        root = tk.Tk()
        root.title("Select Stage")
        root.geometry("256x140")
        tk.Label(root, text="Choose Stage:").pack(pady=5)
        combo = ttk.Combobox(root, values=valid_stages, width=20, state="readonly")
        combo.pack(pady=5)
        combo.current(0)
        combo.focus()
        tk.Button(root, text="OK", command=on_submit).pack(pady=5)
        root.mainloop()
        return selection[0] if selection else None

    stage_str = get_selection()
    if stage_str:
        world, stage = map(int, stage_str.split("-"))
        return (f"SuperMarioBros-{stage_str}-v0", (world, stage))
    return None

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_gray = cv2.resize(gray, (128, 120), interpolation=cv2.INTER_AREA)
    return torch.tensor(resized_gray.copy(), dtype=torch.uint8)  # shape: [120, 128]

def record_expert_data():
    config = load_config()
    selected_stage = select_stage()
    if not selected_stage is None:
        config["env"]["name"] = selected_stage[0]
    else:
        print("Did not select a level. Closing.")
        return

    env = make_env_human(config["env"])
    states = []
    actions = []
    deltas_x = []
    deltas_y = []

    frame_stack = config["env"].get("frame_stack", 4)
    frame_buffer = deque(maxlen=frame_stack)

    pygame.init()
    screen = pygame.display.set_mode((300,300))
    pygame.display.set_caption("Control Window")

    state = env.reset()
    noop_action = 0
    next_state, reward, done_flag, info = env.step(noop_action)
    done = False
    reset_flag = False
    level_finished = False
    print("Recording... Press ESC to stop.")

    clock = pygame.time.Clock()

    # Initialize previous x/y
    info = env.unwrapped._get_info() if hasattr(env.unwrapped, "_get_info") else {}
    prev_x = info.get("x_pos", 0)
    prev_y = info.get("y_pos", 0)

    try:
        while not done:
            raw_frame = env.render(mode="rgb_array")
            resized = cv2.resize(raw_frame, (768, 720), interpolation=cv2.INTER_NEAREST)
            bgr_frame = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            cv2.imshow("Super Mario Bros", bgr_frame)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    elif event.key == pygame.K_r or event.key == pygame.K_F5:
                        reset_flag = True
                        print("Resetting level...")
                    break

            keys = pygame.key.get_pressed()
            action = get_action_from_keys(keys)

            next_state, reward, done_flag, info = env.step(action)
            actions.append(action)

            # Preprocess and stack frames
            preprocessed = preprocess_frame(raw_frame)  # [120, 128]
            frame_buffer.append(preprocessed)
            if len(frame_buffer) < frame_stack:
                # Pad with first frame if not enough frames yet
                while len(frame_buffer) < frame_stack:
                    frame_buffer.appendleft(preprocessed.clone())

            stacked = torch.stack(list(frame_buffer), dim=0)  # [frame_stack, 120, 128]
            states.append(stacked)

            # Calculate deltas
            x_pos = info.get("x_pos", prev_x)
            y_pos = info.get("y_pos", prev_y)
            delta_x = x_pos - prev_x
            delta_y = y_pos - prev_y
            deltas_x.append(delta_x)
            deltas_y.append(delta_y)
            prev_x = x_pos
            prev_y = y_pos

            if info.get("flag_get", 1) or info.get("world", 1) != selected_stage[1][0]:
                level_finished = True
                done = True

            if done_flag and not info.get("flag_get", 1):
                waiting = True
                while waiting and not reset_flag:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_r or event.key == pygame.K_F5:
                                reset_flag = True
                            if event.key == pygame.K_ESCAPE:
                                done = True
                waiting = False
                state = env.reset()
                actions.clear()
                states.clear()
                deltas_x.clear()
                deltas_y.clear()
                frame_buffer.clear()
                reset_flag = False
                continue

            if reset_flag:
                state = env.reset()
                actions.clear()
                states.clear()
                deltas_x.clear()
                deltas_y.clear()
                frame_buffer.clear()
                reset_flag = False
                continue

            state = next_state
            done = done or done_flag
            clock.tick(15)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                done = True

    except KeyboardInterrupt:
        print("Recording interrupted.")

    finally:
        pygame.quit()
        env.close()
        cv2.destroyAllWindows()

        if level_finished:
            world, stage = selected_stage[1]
            stage_folder = f"{world}-{stage}"
            save_dir = os.path.join("data", stage_folder)
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%d%m%y%H%M%S')
            pt_path = os.path.join(save_dir, f"expert_{world}_{stage}_{timestamp}.pt")
            torch.save({
                "states": torch.stack(states),  # [N, frame_stack, 120, 128]
                "actions": torch.tensor(actions),
                "delta_x": torch.tensor(deltas_x, dtype=torch.float32),
                "delta_y": torch.tensor(deltas_y, dtype=torch.float32)
            }, pt_path)
            print(f"Saved expert data to {pt_path}")
        else:
            print("Level not finished. No data saved.")

if __name__ == "__main__":
    record_expert_data()