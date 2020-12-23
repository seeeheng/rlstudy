import matplotlib.pyplot as plt
import numpy as np

def plot_policy(q):
	"""
	Recall that in order to use numpy arrays to store the information, the indexes are one less than the sum.
	i.e., q[0][10] corresponds to a player sum of 1, a dealer first card of 11
	"""
	print("NO USABLE ACE")
	for i in range(10,21):
		q_player = q[i] 
		for j in range(10):
			action = np.argmax(q_player[j][0])
			if action == 0:
				# last_stick[i] = j
				print("For player sum {}, dealer showing {}, policy says to {}".format(i+1,j+1,"Stick"))
			else:
				print("For player sum {}, dealer showing {}, policy says to {}".format(i+1,j+1,"Hit"))
		print()

	print("USABLE ACE")
	for i in range(10,21):
		q_player = q[i] 
		for j in range(10):
			action = np.argmax(q_player[j][1])
			if action == 0:
				# last_stick[i] = j
				print("For player sum {}, dealer showing {}, policy says to {}".format(i+1,j+1,"Stick"))
			else:
				print("For player sum {}, dealer showing {}, policy says to {}".format(i+1,j+1,"Hit"))
		print()
plot_policy(mc.q)
np.save("./q",mc.q)
print(mc.random)
plt.show()
