from game.ia.train import train_agent


env, agent = train_agent(num_episodes=3000000, checkpoint_dir='./model') 