import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 128)

        self.fc_pi = nn.Linear(128, action_dim)
        self.fc_v = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        logits = self.fc_pi(x)
        value = self.fc_v(x)
        return logits, value

    def get_action(self, state):
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), probs.detach()