#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from model import ActorCritic
from utils import state_to_tensor, plot_loss

# Transfers gradients from thread-specific model to shared model
def _transfer_grads_to_shared_model(model, shared_model):
	for param, shared_param in zip(model.parameters(), shared_model.parameters()):
		if shared_param.grad is not None:
			return
		shared_param._grad = param.grad

# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
	for param_group in optimiser.param_groups:
		param_group['lr'] = lr

# Updates networks
def _update_networks(args, T, model, shared_model, loss, optimiser):
	# Zero shared and local grads
	optimiser.zero_grad()
	"""
	Calculate gradients for gradient descent on loss functions
	Note that math comments follow the paper, which is formulated for gradient ascent
	"""
	loss.backward()
	# Gradient L2 normalisation
	nn.utils.clip_grad_norm(model.parameters(), args.max_gradient_norm)

	# Transfer gradients to shared model and update
	_transfer_grads_to_shared_model(model, shared_model)
	optimiser.step()
	if args.lr_decay:
		# Linearly decay learning rate
		_adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T.value()) / args.T_max, 1e-32))

def _train(args, T, model, shared_model, optimiser, policies, Vs, actions, rewards, R):
	action_size = policies[0].size(1)
	policy_loss, value_loss = 0, 0

	# Calculate n-step returns in forward view, stepping backwards from the last state
	t = len(rewards)
	for i in reversed(range(t)):
		# R ← r_i + γR
		R = rewards[i] + args.discount * R
		# Advantage A ← R - V(s_i; θ)
		A = R - Vs[i]

		# Value update dθ ← dθ - ∇θ∙1/2∙(Qret - Q(s_i, a_i; θ))^2d
		value_loss += (A ** 2).mean(0)

		# Log policy log(π(a_i|s_i; θ))
		log_prob = policies[i].gather(1, actions[i]).log()
		# dθ ← dθ + ∇θ∙log(π(a_i|s_i; θ))∙A
		single_step_policy_loss = -(log_prob * A.detach()).mean(0) # Average over batch
		policy_loss += single_step_policy_loss

		# Entropy regularisation dθ ← dθ + β∙∇θH(π(s_i; θ))
		policy_loss -= args.entropy_weight * -(policies[i].log() * policies[i]).sum(1).mean(0)  # Sum over probabilities, average over batch

	# Update networks
	_update_networks(args, T, model, shared_model, policy_loss + value_loss, optimiser)

	return policy_loss, value_loss

# Acts and trains model
def train(rank, args, T, shared_model, optimiser):
	torch.manual_seed(args.seed + rank)

	env = gym.make(args.env)
	env.seed(args.seed + rank)
	model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
	model.train()

	t = 1  # Thread step counter
	epr, eploss, done  = 0, 0, True # Start new episode

	while T.value() <= args.T_max:
		while True:
			model.load_state_dict(shared_model.state_dict()) # sync with shared model
			# Get starting timestep
			t_start = t

			policies, Vs, actions, rewards = [], [], [], [] # save values for computing gradientss

			# Reset or pass on hidden state
			if done:
				hx, avg_hx = Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))
				cx, avg_cx = Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))
				# Reset environment and done flag
				state = state_to_tensor(env.reset())
				done, episode_length = False, 0
			else:
				# Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
				hx = hx.detach()
				cx = cx.detach()

			while not done and t - t_start < args.t_max:
				# Calculate policy and values
				policy, V, (hx, cx) = model(Variable(state), (hx, cx))

				# Sample action
				action = policy.multinomial().data[0, 0]

				# Step
				next_state, reward, done, _ = env.step(action)
				next_state = state_to_tensor(next_state)
				reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
				done = done or episode_length >= args.max_episode_length  # Stop episodes at a max length
				episode_length += 1  # Increase episode counter
								
				# Save outputs for online training
				[arr.append(el) for arr, el in zip((policies, Vs, actions, rewards),
									 (policy, V, Variable(torch.LongTensor([[action]])), Variable(torch.Tensor([[reward]]))))]

				# Increment counters
				t += 1
				T.increment()

				# Update state
				state = next_state

			if done:
				R = Variable(torch.zeros(1, 1))
			else:
				# R = V(s_i; θ) for non-terminal s
				_, R, _ = model(Variable(state), (hx, cx))
				R = R.detach()

			# Train the network on-policy
			p_loss, v_loss = _train(args, T, model, shared_model, optimiser, policies, Vs, actions, rewards, R)

			# Finish episode
			if done:
				break