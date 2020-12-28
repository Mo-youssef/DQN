tensorboard_dir_name = 'test'
env_name = "MiniGrid-DoorAutoKey-16x16-v0"
env_alias = "test"
max_frames = 2 * 60 * 60
max_steps = int(1e7)
dueling = 1
grad_norm = 10
model_copy = 10000
lr = 0.001
per_size = 1e6
per_alpha = 0.5
per_beta = 0.4
update_interval = 4
per_beta_steps = max_steps / update_interval
per_num_steps = 3    # this is the number of steps in the future before using neural network estimate like TD lambda
BS = 32
epsilon_min = 0.02
epsilon_max = 1
explorer_method = 0  # 0 means linear schedule, 1 means random selection NOTE MODIFY EPSILON_MIN AND MAX
epsilon_steps = 5e5
epsilon_num = 256
epsilon_interval = 2000
warmup = 10000
discount = 0.99

ir_beta = 0.1
video_every = 200
ir_warmup = 100

RND_reward = 0
rnd_clip = 0

NGU_reward = 1
ngu_update_interval = 2
ngu_embed_size = 16
ngu_k_neighbors = 10
ngu_L = 5

IR_reward = RND_reward or NGU_reward

rainbow = 1
noisynet = 1
noisy_net_sigma = 0.5
n_atoms = 51
v_max = 10
v_min = -10
