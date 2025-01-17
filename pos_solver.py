#!/usr/bin/env python3

###################################
# CS B551 Fall 2018, Assignment #3
#
# Author: Parth Naik (naikpa)
#
# (Based on skeleton code by D. Crandall)
#
#
####

import random
import math
import operator
import pdb
import numpy
from collections import defaultdict


class Solver:
	def posterior(self, model, sentence, label):
		if model == "Simple":				
			log_probs = 0
			for i in range(len(sentence)):
				if sentence[i] in self.emission[label[i]].keys():
					log_probs += math.log(self.prior[label[i]]) + math.log(self.emission[label[i]][sentence[i]])
				else:
					log_probs += math.log(self.prior[label[i]]) + math.log(0.000001)
			return log_probs
		elif model == "Complex":
			log_probs = 0
			for i in range(len(sentence)):
				word = sentence[i]
				tag = label[i]
				# Include only emission and prior
				if i == 0:
					if word in self.emission[tag].keys():
						log_probs += math.log(self.prior[tag]) + math.log(self.emission[tag][word])
					else:
						log_probs += math.log(self.prior[tag]) + math.log(0.000001)
				# Include transition from prev state
				elif i == 1:
					prev_tag = label[i-1]
				
					if word in self.emission[tag].keys():
						log_probs += math.log(self.emission[tag][word])
					else:
						log_probs += math.log(0.000001)
						
					if tag in self.transition[prev_tag]:
						log_probs += math.log(self.transition[prev_tag][tag])
					else:
						log_probs += math.log(0.000001)
				# Include transition from prev to prev state
				else:
					prev_tag = label[i-1]
					prev_prev_tag = label[i-2]
					
					if word in self.emission[tag].keys():
						log_probs += math.log(self.emission[tag][word])
					else:
						log_probs += math.log(0.000001)
						
					if tag in self.transition[prev_tag]:
						log_probs += math.log(self.transition[prev_tag][tag])
					else:
						log_probs += math.log(0.000001)
						
					if tag in self.transition_next_to_next[prev_prev_tag]:
						log_probs += math.log(self.transition_next_to_next[prev_prev_tag][tag])
					else:
						log_probs += math.log(0.000001)
			return log_probs
		elif model == "HMM":
			return self.hmm_posterior
		else:
			print("Unknown algo!")
			
	# Do the training
	def train(self, lines):
		emission_count = {'adj':{},'adv':{},'adp':{},'conj':{},'det':{},'noun':{},'num':{},'pron':{},'prt':{},'verb':{},'x':{},'.':{}}
		tag_count = {'adj':0,'adv':0,'adp':0,'conj':0,'det':0,'noun':0,'num':0,'pron':0,'prt':0,'verb':0,'x':0,'.':0}
		transition_count = {'adj':{},'adv':{},'adp':{},'conj':{},'det':{},'noun':{},'num':{},'pron':{},'prt':{},'verb':{},'x':{},'.':{}}
		transition_next_to_next_count = {'adj':{},'adv':{},'adp':{},'conj':{},'det':{},'noun':{},'num':{},'pron':{},'prt':{},'verb':{},'x':{},'.':{}}
			
		for line in lines:
			# Get words in the line
			line = line.split(" ")
			# Remove ending spaces and /n characters
			line = line[:-3]
			# Previous tag (required to calculate transition probabilities)
			prev_tag = 0
			# Previous to previous tag (required for MCMC)
			prev_prev_tag = 0
			# Loop through to get emission_count and tag_count
			for i in range(0,len(line),2):
				# Word
				word = line[i].lower()
				# Corresponding tag
				tag = line[i+1].lower()
				
				# Increment the word count in emission_count
				if word not in emission_count[tag].keys():
					emission_count[tag][word] = 1
				else:
					emission_count[tag][word] += 1
					
				# Increment tag_count
				tag_count[tag] += 1
				
				# Increment tag count in transition_count
				if prev_tag!=0:
					if tag not in transition_count[prev_tag].keys():
						transition_count[prev_tag][tag] = 1
					else:
						transition_count[prev_tag][tag] += 1
				
				if prev_prev_tag!=0:
					if tag not in transition_next_to_next_count[prev_tag].keys():
						transition_next_to_next_count[prev_tag][tag] = 1
					else:
						transition_next_to_next_count[prev_tag][tag] += 1
				
				# Set the prev_tag to current tag
				prev_prev_tag = prev_tag
				prev_tag = tag
				
		
		# Calculate prior probabilities
		tag_probs = tag_count.copy()
		total_tag_count = sum(tag_count.values())
		for tag in tag_probs.keys():
			tag_probs[tag] = tag_probs[tag]/total_tag_count
		
		# Calculate emission probabilities
		emission_probs = emission_count.copy()
		for tag in emission_probs.keys():
			for word in emission_probs[tag].keys():
				emission_probs[tag][word] = emission_probs[tag][word]/tag_count[tag]
		
		# Calculate transition probabilities
		transition_probs = transition_count.copy()
		for tag in transition_probs.keys():
			tag_num = sum(transition_probs[tag].values())
			for next_tag in transition_probs[tag].keys():
				transition_probs[tag][next_tag] = transition_probs[tag][next_tag]/tag_count[tag]
				
		# Calculate transition next to next probabilities
		transition_next_to_next_probs = transition_next_to_next_count.copy()
		for tag in transition_next_to_next_probs.keys():
			tag_num = sum(transition_next_to_next_probs[tag].values())
			for next_tag in transition_next_to_next_probs[tag].keys():
				transition_next_to_next_probs[tag][next_tag] = transition_next_to_next_probs[tag][next_tag]/tag_count[tag]
				
		# print(transition_probs['noun']['noun'])
		# input()
		
		# Set the probabilities to local variables
		self.prior = tag_probs
		self.emission = emission_probs
		self.transition = transition_probs
		self.transition_next_to_next = transition_next_to_next_probs
		
		print("Probabilities learnt !")
		# print(self.transition['pron'])
		# input()
		
    # Functions for each algorithm.
	def simplified(self, sentence):
		# Predicted sentence tags
		predicted_sentence = []
		
		for word in sentence:
			temp_dict = {'adj':0,'adv':0,'adp':0,'conj':0,'det':0,'noun':0,'num':0,'pron':0,'prt':0,'verb':0,'x':0,'.':0}
			for tag in temp_dict.keys():
				# Check if word has occured as a emission in the training data for a tag.
				if word in self.emission[tag].keys():
					prob = self.emission[tag][word]
				else:
					prob = 0.0000001
				prob *= self.prior[tag]
				temp_dict[tag] = prob
			# Maximize the product of prior and emission probabilities.
			predicted_sentence.append(max(temp_dict.items(), key=operator.itemgetter(1))[0])
		
		return predicted_sentence
		
		
	def calculate_intersection_probs(self,sentence,p0,selected_rv_position):
		"""Function to calculate intersection probs for MCMC"""
		# Variable to store probability
		prob = 1
		
		# For the first word
		word = sentence[0]
		tag = p0[0]
		
		if word in self.emission[tag].keys():
			prob = prob * self.emission[tag][word] * self.prior[tag]
		else:
			prob = prob * 0.000001 * self.prior[tag]
		
		# For the remaining words upto the previous word
		for i in range(1,selected_rv_position):
			word = sentence[i]
			curr_tag = p0[i]
			prev_tag = p0[i-1]
			
			# Prev to prev transition probability effect only from the third word onwards
			if i > 1:
				prev_prev_tag = p0[i-2]	
						
			# Next to next transition probability effect only till the third last word
			if i < (len(sentence)-3):
				next_next_tag = p0[i+2]
				
			# Next transition probability effect only till the second last word
			if i < (len(sentence)-2):
				next_tag = p0[i+1]
			
			if curr_tag in self.transition[prev_tag].keys():
				prob *= self.transition[prev_tag][curr_tag]
			else:
				prob *= 0.000001
				
			if 'prev_prev_tag' in locals():
				if curr_tag in self.transition_next_to_next[prev_prev_tag].keys():
					prob *= self.transition_next_to_next[prev_prev_tag][curr_tag]
				else:
					prob *= 0.000001
					
			if 'next_tag' in locals():
				if next_tag in self.transition[curr_tag].keys():
					prob *= self.transition[curr_tag][next_tag]
				else:
					prob *= 0.000001
			
			if 'next_next_tag' in locals():
				if next_next_tag in self.transition_next_to_next[curr_tag].keys():
					prob *= self.transition_next_to_next[curr_tag][next_next_tag]
				else:
					prob *= 0.000001
			
			if word in self.emission[curr_tag].keys():
				prob *= self.emission[curr_tag][word]
			else:
				prob *= 0.000001
				
		return prob
			
		
	def sample(self, dict_prob):
		"""Function samples and returns a tag given the probabilities of the tags"""
		# Make sure the probabilities sum to 1.
		probs = numpy.array(list(dict_prob.values()))
		probs /= probs.sum()
		# Return the sampled tag
		return numpy.random.choice(a=list(dict_prob.keys()),size=1,p=probs)
		
	def complex_mcmc(self, sentence):
		# List of tags
		tags = ['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
		
		# The first particle
		p0 = ['noun'] * len(sentence)
		
		burnin = 100
		number_of_particles = 100
		collected_particles = []
		
		# Start burnin
		for i in range(burnin+number_of_particles):
			# Select a RV to sample while keeping all the others the same
			for particle_no in range(len(sentence)):
				selected_rv_position = particle_no
				word = sentence[selected_rv_position]
				# Dictionary to store probs p(X_i|X_1-i)
				dict_prob = {}
				# For the first word
				if selected_rv_position == 0:
					# Because there is a single word sentence!
					if len(sentence) > 1:
						next_tag = p0[1]
					# Calculate probability for each tag
					for tag in tags:
						if sentence[selected_rv_position] in self.emission[tag].keys():
							prob = self.emission[tag][word] * self.prior[tag]
						else:
							prob = 0.000001 * self.prior[tag]
						
						if 'next_tag' in locals():
							if next_tag in self.transition[p0[0]].keys():
								prob *= self.transition[p0[0]][next_tag]
							else:
								prob *= 0.000001
						dict_prob[tag] = prob
				else:
					# Dictionary to store probs p(X_i|X_1-i)
					dict_prob = {}
					for tag in tags:
						prob = self.calculate_intersection_probs(sentence,p0,selected_rv_position)
						
						# Incorporate the transition and emission for all the possible tags for the selected RV.
						prev_tag = p0[selected_rv_position-1]
						curr_tag = tag
						
						if curr_tag in self.transition[prev_tag].keys():
							prob *= self.transition[prev_tag][curr_tag]
						else:
							prob *= 0.000001
			
						if word in self.emission[curr_tag].keys():
							prob *= self.emission[curr_tag][word]
						else:
							prob *= 0.000001
							
						dict_prob[tag] = prob
				# Change the selected tag to the recieved sample	
				p0[selected_rv_position] = self.sample(dict_prob)[0]
			
			# Append the particle to the list of collected particles.
			collected_particles.append(p0.copy())
				
		# Throw away particles collected during burnin
		collected_particles = collected_particles[burnin:]
		
		# pdb.set_trace()
		
		# Find probabilities from the collected_particles for each word for each tag
		probs_from_samples = defaultdict(dict)
		
		for i in range(len(sentence)):
			word = sentence[i]
			for tag in tags:
				p_tag = len([x for x in collected_particles if x[i]==tag])/len(collected_particles)
				probs_from_samples[word][tag] = p_tag
		
		# Now assign tags to words by maximizing observed probabilities
		predictions = []
		
		for word in sentence:
			predictions.append(max(probs_from_samples[word].items(), key=operator.itemgetter(1))[0])
		
		return predictions
    
	def hmm_viterbi(self, sentence):
		# [[(),()]]
		# pdb.set_trace()
		
		# List of tags
		tags = ['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
		
		# Viterbi table
		v_table = []
		
		# Initial start for Viterbi
		# First word
		word0 = sentence[0]
		
		# List of probs for various states
		v_list = []
		for tag in tags:
			# Calculate probs of being in a state
			if word0 in self.emission[tag].keys():
				v_tag = math.log(self.emission[tag][word0]) #+ math.log(self.prior[tag])
			else:
				v_tag = math.log(0.000001) #+ math.log(self.prior[tag])
			# Append as a tuple (prob,state)
			v_list.append((v_tag,tag,tag))
		# Append the probs of the first time instant to the table
		v_table.append(v_list)
		
		# Viterbi for the next n-1 states
		for i in range(1,len(sentence)):
			word = sentence[i]
			# List of probs for various states
			v_list = []
			# Calculate v values for the present 12 possible states
			for tag_curr in tags:
				# Calculate list of product of prob of prev state and transition probabilities
				v_prod_t = []
				for j in range(0,len(tags)):
					tag_old = v_table[i-1][j][2]
					if tag_curr in self.transition[tag_old].keys():
						v_prod_t.append((v_table[i-1][j][0]+math.log(self.transition[tag_old][tag_curr]),tag_old,tag_curr))
					else:
						v_prod_t.append((v_table[i-1][j][0]+math.log(0.000001),tag_old,tag_curr))
				
				# Calculate probs of being in a state
				if word in self.emission[tag_curr].keys():
					v_tag = math.log(self.emission[tag_curr][word]) + max(v_prod_t, key=operator.itemgetter(0))[0] #+ math.log(self.prior[tag_curr])"""
				else:
					v_tag = math.log(0.0000001) + max(v_prod_t, key=operator.itemgetter(0))[0] #+ math.log(self.prior[tag_curr])"""
				# Append as a tuple (prob,state)
				v_list.append((v_tag,max(v_prod_t, key=operator.itemgetter(0))[1],tag_curr))
			# Append the probs of the first time instant to the table
			v_table.append(v_list)
		
		#print(sentence)
		
		#for line in v_table:
		#	print(line,'\n')
		#input()
		# Viterbi table has been created
		
		# List to store the predictions
		predictions = []
		
		# Previous tag
		max_prob = max(v_table[len(sentence)-1], key=operator.itemgetter(0))
		predictions.append(max_prob[2])
		prev_tag = max_prob[1]
		# The posterior of MCMC
		self.hmm_posterior = max_prob[0]
		
		# Backtrack to get MAP
		for k in range(len(sentence)-2,-1,-1):
			for table_entry in v_table[k]:
				if table_entry[2] == prev_tag:
					predictions.append(table_entry[2])
					prev_tag = table_entry[1]
					break
	
		#print(sentence)
		
		#for line in v_table:
		#	print(line,'\n')
		#input()
		
		#print(predictions[::-1])
		#input()
	
		return predictions[::-1]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
	
	def solve(self, model, sentence):
		if model == "Simple":
			return self.simplified(sentence)
		elif model == "Complex":
			return self.complex_mcmc(sentence)
		elif model == "HMM":
			return self.hmm_viterbi(sentence)
		else:
			print("Unknown algo!")
