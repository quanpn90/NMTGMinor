import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottle(nn.Module):
	
		def __init__(self, m):
	
			super(Bottle, self).__init__()
			
			self.module = m() # we initialize the module using lambda 
			
		def forward(self, input):
			
			# input should be a 3D variable
			# B x L x H
			B = input.size(0)
			L = input.size(1)
			
			resized2D = input.view(B * L, -1)
			
			output = self.module(resized2D)
			
			resizedOut = output.contiguous().view(B, L, -1)
			
			return resizedOut
