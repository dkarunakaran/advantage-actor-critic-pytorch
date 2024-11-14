import torch
import torch.optim as optim
import gymnasium as gym 
from actor import Actor
from critic import Critic
import yaml
from utility import logger_helper
from torch.distributions import Categorical


# Ref: https://github.com/nikhilbarhate99/Actor-Critic-PyTorch
# Ref: https://github.com/nikhilbarhate99/Actor-Critic-PyTorch/blob/master/model.py

class Train:
    def __init__(self):
        self.random_seed = 543
        self.env = gym.make('CartPole-v1')
        observation, info = self.env.reset()
        with open("config.yaml") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        self.logger = logger_helper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.n)
        self.critic = Critic(self.env.observation_space.shape[0])

    def run(self):
        torch.manual_seed(self.cfg['train']['random_seed'])
        optim_1 = optim.Adam(self.actor.parameters(), lr=self.cfg['train']['lr'], betas=self.cfg['train']['betas'])
        optim_2 = optim.Adam(self.critic.parameters(), lr=self.cfg['train']['lr'], betas=self.cfg['train']['betas'])
        for episode in range(self.cfg['train']['n_epidode']):
            rewards = []
            state = self.env.reset()
            # Converted to tensor
            state = torch.FloatTensor(state[0])
            self.logger.info(f"--------Episode: {episode} started----------")
            # loop through timesteps
            for t in range(self.cfg['train']['n_timesteps']):
                # The actor layer output the action probability as the actor NN has softmax in the output layer
                action_prob = self.actor(state)
                # categorical function can  give categorical distribution from softmax probability or from logits(no softmax layer in output) with logits as attribute 
                action_dist= Categorical(action_prob)

                # Sample the action
                action = action_dist.sample()

                # As we know we do not use categorical cross entropy loss function directly, but contruct manually to have more control.
                # Categorical cross entropy loss function in pytorch does logits to probability using softmax to categorical distribution,
                # then compute the loss. So normally no need to add softmax function to the NN explicilty. In this work we add the softmax layer on the 
                # NN and compute the categorical distribution.

                # Action has to convert from tensor to numpy for env to process
                next_state, reward, done, _, _= self.env.step(action.detach().numpy())
                
                # Get the log probability to get log pi_theta(a|s)
                log_prob = action_dist.log_prob(action).unsqueeze(0)
                
                # Compute the current state-value 
                v_st = self.critic(state)
                
                # Compute the state-value for next state
                next_state = torch.FloatTensor(next_state)
                v_st_plus_1 = self.critic(next_state)
                

                
            

        

if __name__ == '__main__':
    train = Train()
    train.run()