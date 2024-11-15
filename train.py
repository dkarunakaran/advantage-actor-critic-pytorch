import torch
import torch.optim as optim
import gymnasium as gym 
from actor import Actor
from critic import Critic
import yaml
from utility import logger_helper
from torch.distributions import Categorical
import numpy as np


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
        critic_losses = []
        for episode in range(self.cfg['train']['n_epidode']):
            rewards = []
            log_probs = []
            advantages = []

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

                # Action has to convert from tensor to numpy for env to process
                next_state, reward, done, _, _= self.env.step(action.detach().numpy())
                
                # Get the log probability to get log pi_theta(a|s)
                log_prob = action_dist.log_prob(action).unsqueeze(0)
                
                # Compute the current state-value 
                v_st = self.critic(state)
                
                # Compute the state-value for next state
                next_state = torch.FloatTensor(next_state)
                v_st_plus_1 = self.critic(next_state)

                # Get the advantage from r(s_t,a_t)+v_s_t+1 - v_s_t
                adv = reward+(self.cfg['train']['gamma']*v_st_plus_1.detach().numpy().item()) - v_st.detach().numpy().item()

                # String the value for loss computation
                rewards.append(reward)
                log_probs.append(log_prob)
                advantages.append(adv)

                #assign next state as current state
                state = next_state

                if done:
                    self.logger.info('Iteration: {}, Score: {}'.format(episode, i))
                    break
            
            avg_reward.append(np.mean(rewards))        
            advantages = torch.tensor(advantages)
            rewards = torch.tensor(rewards)
            log_probs = torch.cat(log_probs)

            # Same as cross entropy loss for where cross entropy loss is multiplied with Advantage function. formuala:  sum of log pi_theta(a_t, s_t) * Advantage
            actor_loss = -(log_probs * advantages.detach()).mean()
            actor_losses.append(actor_loss.detach().numpy().item())

            # Some form of advantage function is used to commpute the loss of critic
            critic_loss = advantages.pow(2).mean()
            
            # Need to make required grad true to compute the loss
            critic_loss.requires_grad=True
            critic_losses.append(critic_loss.detach().numpy().item())

            # Compute gradient
            actor_loss.backward()
            critic_loss.backward()

            # optimisation
            actor_optim.step()
            critic_optim.step()

        print(actor_losses)
        torch.save(self.actor, 'model/actor.pkl')
        torch.save(self.critic, 'model/critic.pkl')
        self.env.close()



                

                
            

        

if __name__ == '__main__':
    train = Train()
    train.run()