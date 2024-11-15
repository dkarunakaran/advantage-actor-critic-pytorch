import torch
import torch.optim as optim
import gymnasium as gym 
from actor import Actor
from critic import Critic
import yaml
from utility import logger_helper
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Ref: https://github.com/nikhilbarhate99/Actor-Critic-PyTorch
# Ref: https://github.com/Lucasc-99/Actor-Critic/blob/master/src/a2c.py

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
        actor_optim = optim.Adam(self.actor.parameters(), lr=self.cfg['train']['lr'], betas=self.cfg['train']['betas'])
        critic_optim = optim.Adam(self.critic.parameters(), lr=self.cfg['train']['lr'], betas=self.cfg['train']['betas'])
        avg_reward = []
        actor_losses = []
        avg_actor_losses = []
        critic_losses = []
        avg_critic_losses = []
        eps = np.finfo(np.float32).eps.item()
        for episode in range(self.cfg['train']['n_epidode']):
            rewards = []
            log_probs = []
            state_values = []

            state = self.env.reset()
            # Converted to tensor
            state = torch.FloatTensor(state[0])
            self.logger.info(f"--------Episode: {episode} started----------")
            actor_optim.zero_grad()
            critic_optim.zero_grad()

            # loop through timesteps
            for i in range(self.cfg['train']['n_timesteps']):
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
                
                # Get the log probability to get log pi_theta(a|s) and save it to a list.
                log_probs.append(action_dist.log_prob(action))
                
                # Compute the current state-value 
                v_st = self.critic(state)
                state_values.append(v_st)

                # Action has to convert from tensor to numpy for env to process
                next_state, reward, done, _, _= self.env.step(action.detach().numpy())
                rewards.append(reward)

                # Assign next state as current state
                state = torch.FloatTensor(next_state) 

                # Enviornment return done == true if the current episode is terminated
                if done:
                    self.logger.info('Iteration: {}, Score: {}'.format(episode, i))
                    break
            
            R = 0
            actor_loss_list = [] # list to save actor (policy) loss
            critic_loss_list = [] # list to save critic (value) loss
            returns = [] # list to save the true values

            # Calculate the return of each episode using rewards returned from the environment in the episode
            for r in rewards[::-1]:
                # Calculate the discounted value
                R = r + self.cfg['train']['gamma'] * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)
            for log_prob, state_value, R in zip(log_probs, state_values, returns):

                # Advantage is calculated by the difference between actual return of current stateand estimated return of current state(v_st)
                advantage = R - state_value.item()

                # Calculate actor (policy) loss
                a_loss = -log_prob * advantage
                actor_loss_list.append(a_loss) # Instead of -log_prob * advantage

                # Calculate critic (value) loss using huber loss
                # Huber loss, which is less sensitive to outliers in data than squared-error loss. In value based RL ssetup, huber loss is preferred.
                # Smooth L1 loss is closely related to HuberLoss
                c_loss =  F.smooth_l1_loss(state_value, torch.tensor([R])) #F.huber_loss(state_value, torch.tensor([R]))
                critic_loss_list.append(c_loss)

            # Sum up all the values of actor_losses(policy_losses) and critic_loss(value_losses)
            actor_loss = torch.stack(actor_loss_list).sum()
            critic_loss = torch.stack(critic_loss_list).sum()

            # Perform backprop
            actor_loss.backward()
            critic_loss.backward()
            
            # Perform optimization
            actor_optim.step()
            critic_optim.step()

            # Storing average losses for plotting
            if episode%50 == 0:
                avg_actor_losses.append(np.mean(actor_losses))
                avg_critic_losses.append(np.mean(critic_losses))
                actor_losses = []
                critic_losses = []
            else:
                actor_losses.append(actor_loss.detach().numpy())
                critic_losses.append((critic_loss.detach().numpy()))

        plt.figure(figsize=(10,6))
        plt.xlabel("X-axis")  # add X-axis label
        plt.ylabel("Y-axis")  # add Y-axis label
        plt.title("Average actor loss")  # add title
        plt.savefig('actor_loss.png')
        plt.plot(avg_actor_losses)
        plt.close()

        plt.figure(figsize=(10,6))
        plt.xlabel("X-axis")  # add X-axis label
        plt.ylabel("Y-axis")  # add Y-axis label
        plt.title("Average critic loss")  # add title
        plt.plot(avg_critic_losses)
        plt.savefig('critic_loss.png')
        plt.close()

        torch.save(self.actor, 'model/actor.pkl')
        torch.save(self.critic, 'model/critic.pkl')
        self.env.close()



                

                
            

        

if __name__ == '__main__':
    train = Train()
    train.run()