import os
import json
import yaml
import torch
import pygame
import tkinter as tk
import numpy as np
import cv2
import datetime
from tkinter import simpledialog
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
    root = tk.Tk()
    root.withdraw()
    input_stage = simpledialog.askstring("Select Stage", "Enter the stage (e.g., 1-1):")
    root.destroy()
    if input_stage:
        return (f"SuperMarioBros-{input_stage}-v0",(int(input_stage[0]), int(input_stage[2])))
    return None

def record_expert_data():
    config = load_config()
    selected_stage = select_stage()
    if selected_stage:
        config["env"]["name"] = selected_stage[0]

    env = make_env_human(config["env"])
    expert_data = []
    
    pygame.init()
    screen = pygame.display.set_mode((300,300))

    expert_actions = []
    state = env.reset()
    done = False
    reset_flag = False
    level_finished = False
    print("Recording... Press ESC to stop.")

    clock = pygame.time.Clock()

    try:
        while not done:
            raw_frame = env.render(mode="rgb_array")
            resized = cv2.resize(raw_frame, (512, 480), interpolation=cv2.INTER_NEAREST)
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
            expert_actions.append(action)

            if info.get("flag_get") or info.get("world", 1) != selected_stage[1][0]:
                level_finished = True
                done = True

            if reset_flag or info.get("life") == 0:
                state = env.reset()
                expert_actions.clear()
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
        expert_data.append({
            "actions": expert_actions
        })
        if level_finished:
            os.makedirs(os.path.dirname(config["data"]["expert_data_path"]), exist_ok=True)
            json_path = os.path.splitext(config["data"]["expert_data_path"])[0] + "-" + str(selected_stage[1][0])+"-"+ str(selected_stage[1][0]) + "_" + datetime.datetime.now().strftime("%d%m%y") + ".json"
            with open(json_path, "w") as file:
                json.dump(expert_data, file)
            print(f"Saved expert data to {json_path}")
        else:
            print("Level not finished. No data saved.")


if __name__ == "__main__":
    record_expert_data()
