import numpy as np

# SIMPLE BANDIT ALGORITHM
# Q(A) <- Q(A) 1/ N(A) [ R - Q(A)]

agent = ten_armed_agent()
for i in range(10000):
	agent.step()
	agent.show_q()