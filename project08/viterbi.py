# Justin, Linh, Hongyuan
# Project 8: Viterbi Algorithm

# import statements
import numpy as np
from ..EmissionSet_and_HiddenState_defs_linh import EmissionSet, HiddenStates
from ..HMModel_def import HMModel


def identify_emissionset(emission_seq: list[str], model: HMModel, verbose: bool = False) -> EmissionSet | None:
    """
    Function to identify the name for the emission set that should be queried
    when looking up the probabilities in the given model for the supplied 
    sequence of emissions

    Parameters
    ----------
    emission_seq : list[str]
        the sequence of emissions we're checking against the model
    model: HMModel
        The hidden markov model whose emission sets are being queried here

    Returns
    -------
    str
        the name of the emission set that should be used to look up 
        probabilities for this sequence of emissions
    returns None if no match is found
    """
    # get the unique emissions in the sequence
    unique_emissions = set(emission_seq)
    if verbose:
        print(f"Unique emissions in sequence: {unique_emissions}")

    # iterate over the model's emission sets
    for curr_emission_set in model.emission_sets:
        # convert the list of unique emission names to a set
        legal_emissions = set(curr_emission_set.value_names)
        if verbose:
            print(f"Current EmissionSet: {curr_emission_set.name}\n"
                f"Legal emissions: {legal_emissions}")

        # see what emissions are in the sequence that aren't in the current emission set
        diff = unique_emissions - legal_emissions
        if verbose:
            print(f"Elements in emission seq that aren't legal: {diff}")
        
        # if they're the same, return the name of the emission set
        if not diff:
            return curr_emission_set
    
    # if we make it out here, there's no match so return none
    return None


#def lookup_emission_prob(model, emission_set)


def Viterbi(emission_seq: list[str], model: HMModel) -> list[str]:
    """
    Function implementing the Viterbi algorithm to determine the optimal
    sequence of hidden states that could have given rise to the sequence
    of emissions in emission_seq

    Parameters
    ----------
    emission_seq : list[str]
        list of emissions, where each emission is a string of unknown length
    model : HMModel
        the hidden markov model whose initial, state transition, and emission probabilities
        are used to determine the optimal sequence of hidden states

    Returns
    -------
    list[str]
        sequence equal in length to the emission_seq, containing the hidden state at each emission
    """
    # attempt to get the name for the emission set we should query for this seq of emissions
    curr_emission_set = identify_emissionset(emission_seq, model, verbose=True)
    if curr_emission_set is None:
        print(f"This sequence of emissions contains a combination of emissions that is not used by any emission sets in {model}")
        return []

    # convert weights in the model to probabilities
    model.normalize_all()

    # make a list of the states in the model
    states = [state for state in model.hidden_states]

    # make a scoring and traceback matrix for dynamic programming
    scores = np.full((len(states), len(emission_seq)), -np.inf)
    traceback = np.empty((len(states), len(emission_seq)), dtype=int)

    # for the first emission, we need to use the initial probs instead of transition probs
    curr_emission = emission_seq[0]
    # get the index for this emission in the current state's emission probabilities
    emission_idx = curr_emission_set.value_names.index(curr_emission)

    # iterate over the states
    for state_idx, curr_state in enumerate(states):
        # the model's initial prob's table is indexed the same way as the list of states
        scores[curr_state, 0] = model.P_init[curr_state.name] * model.P_eh[curr_emission_set.set_name][curr_state.name][emission_idx]
    
    # now we handle every single emission beyond the first
    for i, curr_emission in enumerate(emission_seq[1:]):
        # get the index for this emission in the current state's emission probabilities
        emission_idx = curr_emission_set.value_names.index(curr_emission)

        # iterate through the states we could be at this timepoint
        for state_idx, curr_state in enumerate(states):
            # initialize the current best score and current best previous score
            best_score = scores[state_idx, i]
            best_prev = state_idx
            # get the current emission probability
            emit_prob = model.P_eh[curr_emission_set.set_name][curr_state.name][emission_idx]

            # iterate through the states the PREVIOUS EMISSION could have been on
            for prev_idx, prev_state in enumerate(states):
                # grab the previous state's score
                prev_score = scores[prev_idx, i-1]
                transition_prob = model.P_hh[prev_state.name][curr_state.name]
                # get the current state's score if its traceback is from prev_state
                curr_score = prev_score * transition_prob * emit_prob

                # check if this candidate score is greater than best_score and update accordingly
                if curr_score > best_score:
                    best_score = curr_score
                    best_prev = prev_idx
            
            # write the best score and its traceback to the matrices
            scores[state_idx, i] = best_score
            traceback[state_idx, i] = best_prev

    # TODO: traceback

if __name__ == "__main__":
    # TODO: do something with a toy example for demo purposes here
    print(Viterbi())