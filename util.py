import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray

def preprocess(img):
  re_img = resize(img, (84, 84))
  out = np.array(rgb2gray(re_img)).astype(np.float32)
  return out

def skip_action(action, env, skip_steps):
    reward = 0
    obs = []
    for _ in range(skip_steps):
      state, rew, done, _ = env.step(action)
      reward += rew
      obs.append(preprocess(state))
    obs = np.array(obs)[None, ...]
    return obs, reward, done
