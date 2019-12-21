# POS_Tagging
Part-Of-Speech tagging using probabilistic approaches

Natural language processing (NLP) is an important research area in artificial intelligence, dating back to at least the 1950’s. One of the most basic problems in NLP is part-of-speech tagging, in which the goal is to mark every word in a sentence with its part of speech (noun, verb, adjective, etc.). This is a first step towards extracting semantics from natural language text. For example, consider the following sentence:Her position covers a number of daily tasks common to any social director. Part-of-speech tagging here is not easy because many of these words can take on different parts of speech depending on context. For example, position can be a noun (as in the above sentence) or a verb (as in “They position themselves near the exit”). In fact, covers, number, and tasks can all be used as either nouns or verbs, while social and common can be nouns or adjectives, and daily can be an adjective, noun, or adverb. The correct labeling for the above sentence is: 
Her position covers a number of daily tasks common to any social director. 
DET NOUN VERB DET NOUN ADP ADJ NOUN ADJ ADP DET ADJ NOUN
Where DET stands for a determiner, ADP is an adposition, ADJ is an adjective, and ADV is an adverb.1 Labeling parts of speech thus involves an understanding of the intended meaning of the words in the sentence, as well as the relationships between the words.

<b>To use the 3 models for POS tagging and seeing the results, run using command - python ./label.py bc.train bc.test.tiny<\b><br>

Training phase : 
1.	During the training phase using the tagged words in the training data we calculate the prior, emission and transition probabilities.
	a.	The prior probabilities is the count of a tag divided by the total number of tags/words. It is stored as a dictionary where the keys are the tags
		an the values are the priors of the tags.
	b.	The emission probabilities are the probabilities of observing word w given a tag t i.e P(w|t). The emission probabilities are  stored as a dictionary
		where the keys are the tags, the values are in turn dictionaries whose keys are the words and their values are the probabilities.
		i.e Probability of getting John given noun is self.emission[noun][john]
	c.	The transition probabilities are the probabilities the next tag will be t+1 given the current tag i.e P(t+1,t). The transition probabilities are
		also stored as a dictionary where self.transition[old_tag][new_tag] gives us the transition probability from the old to new tag.
Simple model:
1.	After calculating the probabilities during the training phase we now use the simple model to tag the words of the sentence.
2.	The simple model treats each word individually, that is in the simple model the main aim is to maximize the product of the prior and emission probabilities
	for a word. - P(tag)*P(word|tag)
3.	We choose the tag for the product of the prior and transition probabilities is maximum.
4.	Thus we get the predicted sentence.
5.	Performance : We can see that for only words the accuracy of the model is good but less in case of sentences as compared to other models as the model 
	doesn't take into consideration neighbouring words.
	
Hidden Makrov Model:
1.	In the hidden markov model we find the Maximum A Posteriori sequence i.e the MAP sequence.
2.	In the HMM we find the MAP using the Viterbi algorithm.
3.	We start with the initial time state(word) where for each of the 12 tags we calculate the probabilities as the product of the priors and the emission
	probabilities.
4.	For the next time states(words) for each of the 12 tags we calculate the probabilities as a product of the emission of the word given the state and the 
	maximum of list of products of the 12 state probabilities in the previous time step and the transition probabilities from the old to the new state.
5.	We store these results in a Viterbi table which has the following form - [[(0.0023,noun,adv),(0.113,adj,det),...],[....],[....],[....]] where each
	inside list is for a timestep and the 12 tuples within this list have the first value as the probability, the second value as the old tag and the third
	value is the current tag.
6.	After reaching the last timestep we choose the state tuple having the maximum probability and add the current state to the list of predictions.
7.	Then we backtrack our way back till the first word by finding tuples in the previous state whose current tag is equal to the old_tag in the current state.
8.	Finally we reverse the predictions list and return it.
9.	Performance : The HMM gives the max performance of the three models on the words as well as the sentences, this is because it takes into consideration
	the words on the previous time step.
	
MCMC:
1.	In MCMC or Gibbs sampling we sample from a markov chain which represents the stationary distribution of the desired Bayes Net.
2.	We use MCMC in the case where direct inference is NP hard.
3.	In POS tagging we start with a initial particle having all nouns then we choose one tag and sample over it while keeping all other tags constant i.each
	P(tag_n|all other tags unchanged).
4.	By using variable elimination we find that a central tag's probability depends upon the previous to previous, previous, emission of current, next and 
	next to next tags.
5.	We do this for all tags in the particle to get a new particle.
6.	We collect this particle(if its after the burnin time).
7.	Using the collected particles for each position/word we build a probability distribution taking into consideration the frequency a tag appears for that word
	in our collected particles.
8.	As we collect more and more particles, the closer we get to the actual distribution.
9.	To run the model fairly quickly I have kept the number of particles to be 100 as we increase the number of particles to be collected, the accuracy increases.
10.	Performance : For sufficient number of particles MCMC should theoretically give the best results but for a low particle number of 100, it gives comparable
	results to the simple model.
	
Problems faced:
1.	If a word is not observed in the training data for a tag (emission) and it is encountered in the testing data then logically the emission of the word given
	the tag should be 0 but i practice we cannot assign it 0 (explanation below).
2.	Spent 2 days trying to debug Viterbi code which was constructing a correct Viterbi table but not giving correct predictions, there was a error in backtracking.
	Used pdb - python debugger to set breakpoints and examine variables.
	
Laplace smoothing:
While calculating likelihood for a word i.e P(Word|City) for a word that has not yet occured in the city's tweets the P(Word|City) = 0 seems logical,
but we need to assign some value however small to the probability such that it is not 0 as there is always a very remote possibilty that the word might
occur in the city's tweets.
Also if we assign 0 the probabilty the whole product will be 0 which is not at all desirable.
Results:
                Words correct:     Sentences correct:
   0. Ground truth:      100.00%              100.00%
         1. Simple:       93.71%               42.35%
            2. HMM:       95.03%               50.20%
        3. Complex:       93.76%               44.71%
