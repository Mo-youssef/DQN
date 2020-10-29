import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

env = gym.make('MontezumaRevengeNoFrameskip-v4', frameskip=(2, 3))
fig, ax=plt.subplots()
env.reset()
frames = [] 
for i in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    title = ax.text(0.5,1.05,"Title {}".format(i), 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes, )
    frames.append([plt.imshow(observation,animated=True), title])
ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, 
                                repeat_delay=1000)
plt.axis('off')
ani.save('video/movie.gif')
plt.figure(figsize=(20, 14))
plt.plot([0, 3, 5, 9, 11], '-o')
plt.title('Intrinsic Reward')
plt.xlabel('step')
plt.ylabel('IR')
plt.savefig('video/books_read.png')
env.close()