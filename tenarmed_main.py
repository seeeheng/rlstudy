import numpy as np
from ten_armed_bandit.ten_armed_agent import TenArmedAgent
from ten_armed_bandit.ten_armed_bandit import TenArmedBandit

# SIMPLE BANDIT ALGORITHM
# Q(A) <- Q(A) 1/ N(A) [ R - Q(A)]

agent = TenArmedAgent(TenArmedBandit())
for i in range(10000):
	agent.step()
	agent.show_q()