import json
import os
import time
from decimal import Decimal
import pygame
import gym
import numpy as np
import gym_donkeycar
from PIL import Image

from pretrainer.vae import VAE

DONKEY_GYM = True
DONKEY_SIM_PATH = '/Users/kawamuramasato/Desktop/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim'#"/Applications/donkey_sim.app/Contents/MacOS/donkey_sim"
DONKEY_GYM_ENV_NAME = 'donkey-circuit-launch-track-v0'#"donkey-generated-track-v0"

agent_min_throttle = 0.2
agent_max_throttle = 0.5
agent_max_steering_diff = 0.35
agent_max_steering = 1.0
agent_min_steering = -1.0

action_history = [0.0, 0.0] * 10
n_command_history = 20


# REFERENCE https://take6shin-tech-diary.com/usb-controller-pygame/
def main():

    action_processor = ActionPreProcesser(n_command_history,
                                          agent_min_throttle,
                                          agent_max_throttle,
                                          agent_max_steering_diff,
                                          agent_min_steering,
                                          agent_max_steering)
    game_pad = GamePad()
    recorder = Recorder()
    proess_loop = ProcessLoop(game_pad, recorder, action_processor)
    try:
        proess_loop()
    except(KeyboardInterrupt, SystemExit):  # Exit with Ctrl-C
        print("Exit")
        proess_loop.env.close()


class ProcessLoop():

    def __init__(self, game_pad, recorder, action_processor):
        conf = {"exe_path": DONKEY_SIM_PATH, "port": 9091}
        self.env = gym.make(DONKEY_GYM_ENV_NAME, conf=conf)
        self.game_pad = game_pad
        self.recorder = recorder
        self.action_processor = action_processor

    def __call__(self, *args, **kwargs):
        obs = self.env.reset()
        while True:
            recording, steering, throttle = self.game_pad.get_state()
            p_action = [steering, throttle]
            action = self.action_processor(p_action)
            obs, reward, done, info = self.env.step(action)
            if recording == 1:
                self.recorder.record(obs, p_action, self.action_processor.action_history)
            time.sleep(0.04)


class GamePad():

    def __init__(self):
        pygame.init()
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        pygame.event.get()

    def get_state(self):
        recording = self.joy.get_button(5)
        throttle = -1.0 * float(Decimal(self.joy.get_axis(1)).quantize(Decimal('0.01')))
        steering = float(Decimal(self.joy.get_axis(2)).quantize(Decimal('0.01')))
        pygame.event.pump()
        pygame.event.get()
        return recording, steering, throttle


class Recorder():

    def __init__(self, folder_name='dataset', prefix='record'):
        self.record_num = 0
        self.prefix = prefix
        self.folder_name = folder_name
        os.mkdir(folder_name)

    def record(self, image, action, history):
        self._save(image, action, history)
        if self.record_num % 100 == 0:
            print('Recording: {}'.format(self.record_num))

    def _save(self, image, action, history):
        p = self._save_image(image)
        self._save_telemetry(history, action, p)
        self.record_num += 1

    def _save_image(self, image):
        file_path = os.path.join(self.folder_name, 'cam_{}.jpg'.format(self.record_num))
        img = Image.fromarray(np.uint8(image))
        img.save(file_path)
        return file_path

    def _save_telemetry(self, history, action, image_file_path):
        telemetry = {
            'user/angle': action[0],
            'user/throttle': action[1],
            'history': history,
            'image': image_file_path
        }
        file_path = os.path.join(self.folder_name, '{}_{}.json'.format(self.prefix, self.record_num))
        with open(file_path, 'x') as fp:
            json.dump(telemetry, fp=fp)


class ActionPreProcesser():

    def __init__(self, n_command_history, agent_min_throttle, agent_max_throttle, agent_max_steering_diff,
                 agent_min_steering, agent_max_steering):
        self.n_command_history = n_command_history
        self.agent_min_throttle = agent_min_throttle
        self.agent_max_throttle = agent_max_throttle
        self.agent_max_steering_diff = agent_max_steering_diff
        self.agent_min_steering = agent_min_steering
        self.agent_max_steering = agent_max_steering
        self.action_history = [0.0, 0.0] * (n_command_history * 2)

    def _scaled_action(self, action):
        # Convert from [-1, 1] to [0, 1]
        t = (action[1] + 1) / 2
        action[1] = (1 - t) * self.agent_min_throttle + self.agent_max_throttle * t
        return action

    def _smoothing_action(self, action):
        if n_command_history > 0:
            prev_steering = action_history[-2]
            max_diff = (self.agent_max_steering_diff - 1e-5) * (
                    self.agent_max_steering - self.agent_min_steering)
            diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
            action[0] = prev_steering + diff
        return action

    def _record_history(self, action):
        if len(action_history) >= self.n_command_history * 2:
            del self.action_history[:2]
        for v in action:
            self.action_history.append(v)

    def __call__(self, action):
        action = self._scaled_action(action)
        action = self._smoothing_action(action)
        self._record_history(action)
        return action


if __name__ == "__main__":
    main()
