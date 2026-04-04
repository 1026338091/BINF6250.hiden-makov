# BINF6250 Project 9:

## Description of the project

## Dependencies

## Contents

## Usage

## Pseudocode

```
Forward Algo:
    Inputs:
        Sequence of emissions
        Model
    output:
        The entire scoring matrix (so we don't have to recalculate things when we want to look up positions later)
        The sum of the last column in the scoring matrix

    1. Make an M x N matrix where N is length of emissions sequence (assume it's a list of list of strings // list of strings)
       and M is the number of states in our model
    2. Compute column 0 using: (remember to convert these probabilities to log space)
        initial probabilities for each hidden state
        Emission probabilities at t=0
    3. Iterate through the emissions in the sequence (enumerate here to access corresponding column number in scoring matrix)
            Skip or ignore the first emission
        4. Iterate through each hidden state
            5. We need to calculate Emission probability in this square P(observation at current t | current hidden state)
                Convert this to logspace
            6. Initialize a sum at 0
            7. Iterate through all of the hidden states at the previous timepoint (t-1)
                8. Look up transition probability from prev_state (@ t-1) to curr_state (@ t), and convert to log
                9. Add the score from this cell (previous state at t) plus the log-probability of the transition plus the current (@ t) emission log-prob
                10.np.logaddexp() this to the sum from step 6 (if we weren't in log space, the log-probabilities above would be raw probabilities that we'd multiply, and then in this step we would just add them to the sum)

            11.Now we have the score for this position in the matrix, write it in
    12.Add up the values in the last column, this is the probability of the sequence given the model
    13.Return the scoring matrix and the probability of the sequence given the model


Backward Algo:
    Inputs:
        Sequence of emissions
        Model
    output:
        The entire scoring matrix (so we don't have to recalculate things when we want to look up positions later)
        The sum of the last column in the scoring matrix

    1. Make an M x N matrix where N is length of emissions sequence (assume it's a list of list of strings // list of strings)
       and M is the number of states in our model
    2. Initialize column N with all 1s
    3. Iterate through the emissions in the sequence from back to front (enumerate here to access corresponding column number in scoring matrix)
            Skip or ignore the last emission
        4. Iterate through each hidden state
            5. Initialize a sum at 0
            6. Iterate through all of the hidden states at the next timepoint (t+1)
                7. Look up transition probability from curr_state (@ t) to next_state (@ t+1), and convert to log
                8. Add the following:
                    score from the cell we're looking forward to (state at t+1)
                    log-probability of the transition plus the current (@ t) emission log-prob
                    log of emission probability for emission at t+1 given the state we're looking forward to at t+1
                9. np.logaddexp() this to the sum from step 6 (if we weren't in log space, the log-probabilities above would be raw probabilities that we'd multiply, and then in this step we would just add them to the sum)

            10.Now we have the score for this position in the matrix, write it in
    10.5. Make a beta column and compute the probabilities of the paths going to the beta column using initial probs instead of transition probs (so do all the above stuff for it)
    11.Add up the values in the first column, this is the probability of the sequence given the model
    12.Return the scoring matrix and the probability of the sequence given the model


Helper func to combine forward and backward probability for current position:
    Takes as input:
        Forwards scoring matrix
        Backwards scoring matrix
        Position (row and column number), representing combination of an emission at time t and being in a given state at that time
        Probability of the sequence given the model (could pass either forward or backwards probs here)
    1. Look up the values in both matrices for the given position
    2. Apply the formula:
        foward_prob(current_position) * backward_prob(current_position) / prob(entire sequence | the model)
        Note that our lookups from the forward and backwards matrices are in logspace, so actually just do forward_score + backward_score - log(prob(entire seq | model))
    3. Return the result of that formula


Posterior-decoding:
    Takes as input:
        Sequence of emissions
        The model
    0. Make an ordered list of the states
    1. Make the forwards matrix and the backwards matrix
    2. Initialize a traceback list
    3. Iterate through emissions in the emissions sequence
        4. Start an empty list for forward-backward probabilities
        5. Insert the forward-backward probability for all states into the list (the indices of this prob-list correlate 1:1 with the indices of the list of states)
        6. Grab the argmax (the index of the state with the highest probability at this emission) from the list of fb-probabilities and add it to the traceback
    7. Could convvert traceback list from being a list of indices to a list of state names for readability
        [state[idx] for idx in traceback] boom easy comprehension <3
    8. Return the list of most likely states

```

## Personal reflections
### Successes

### Strugges

### Individual reflections

## Appendix
