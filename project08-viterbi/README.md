# BINF6250 Project 8: Viterbi Algorithm

## Description of the project
This repository contains work done as part of the BINF6250 Algorithmic Foundations in Bioinformatics course (Northeastern University, Spring 2026).

This one's about the Viterbi algorithm for inferrence of the most likely sequence of hidden states, but mostly we were messing around with our model constructor.

## Contents

- this `README.md`
- `project08.ipynb`

## Pseudocode (for the Viterbi algorithm)

```
n = len(emissions)

states = [all possible hidden states in index order] 

scores = (len(states) x n) matrix of -inf

traceback = (len(states) x n) matrix of nones

for s in states:
	scores[0][s] = init probability of s * P(emission[0]|s)

from i in (emission[1] to emission[n]):

	for curr_s in states:
		best_score = scores[i][curr_s]
		best_prev = traceback[i][curr_s]
		emit_p = P(i|curr_s)
		
		for prev_s in states:
			prior = P(prev_s) = scores[i-1][prev_s]
			trans_p = P(prev_s -> curr_s)
			score = prior * trans_p * emit_p
			
			if score > best_score:
				best_score = score
				best_prev = prev_s
		
		scores[i][curr_s] = best_score
		
		traceback[i][curr_s] = best_prev

reverse_path = [states[argmax(scores[n])]]

for i in emission[n] to emmission[0]:
	
	state_at_i = reverse_path[-1]
	where_from = traceback[i][state_at_i]
	reversed_path.append(where_from)
	
return reversed(reverse_path)
```

In the implementation used in the notebook, we don't keep track of the entire lookup table of scores.

## Personal reflections
### Successes

### Strugges

### Individual reflections
Linh: This project has been interesting as we've all been able to explore the specificities of HMM in particular and of Markovness in general. Implementing the Viterbi algorithm seemed fairly straightforward, so we filled the space with an initiative to design an extensible setup for future use, which we will likely benefit from. We also attempted to generalize the model construction itself for arbitrary order of Markov property, as well as extend HMM in directions such as "additionally, condition state probabilities on previous observations". This latter example is conceptually reasonable but would require major mods to classic HMM if we wanted to stick with HMM structure/methods; nothing unprecedented, but the kind of modeling choices you'd make for a specific purpose. We may just be forgetting that models are limited to adressing certain types of questions and designs, and as these projects are meant to teach us the basic algorithmic toolkit for inferrence with HMMs, we should be wary of trying to depart from its most basic assumptions and definitions for no reason. As usual I appreciate the learning and group discussion that the rabbit hole has wrought. 

## Appendix
