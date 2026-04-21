# import statements
import numpy as np
from collections import defaultdict
from assignment_materials.HMM import BaseHMM, HMM


class profile_HMM(BaseHMM):
    """
    Child of class BaseHMM that is specialized to act as
    a profile HMM

    Attributes
    ----------
    
    """
    def __init__(self, alignment: list[list[str]],
                 alphabet: list[str],
                 gap: str = "-",
                 seed = None, 
                 precision = 2, 
                 tolerance = 1e-10):
        """
        Initializes the profile HMM obejct

        Parameters
        ----------
        alignment : list[list[str]]
            multiple sequence alignment formatted as list of list of strings
        alphabet : list[str]
            list of legal emissions
        gap : str, optional
            character used in the multiple sequence alignment to represent a gap in one seq
            by default "-"
        """

        # call helper funcs to define emission and trans probs as np arrays
        trans_probs, emit_probs = self.get_profile_hmm_probs(alignment, alphabet, gap)

        # this list represents the order that match, insertion, and deletion states are written into the 3d arrays
        self.namelist = ["M", "I", "D"]

        # add the gap character to the alphabet so Viterbi and others don't break later
        alphabet = alphabet + [gap]

        # use the shape of our emission probs array to know how many match positions we have (aka how many 
        # profile positions to have). 
        # The idices of the outer array on both trans_probs and emit_probs represent the match positions
        model_length = emit_probs.shape[0]
        self.hidden_states = self._make_states_list(model_length)

        # conver the np arrays to dicts
        trans_probs = self._trans_probs_to_dict(trans_probs)
        emit_probs = self._emit_probs_to_dict(emit_probs, alphabet, gap)

        # make a dict for init probs (we're putting M0 on the transition matrix so this is easier)
        init_probs = defaultdict(float)
        init_probs["M0"] = 1.0

        # now call the initialization code from the parent class so our other attributes are up and running
        super().__init__("".join(alphabet), self.hidden_states, init_probs, 
                 trans_probs, emit_probs, seed=seed, precision=precision, tolerance=tolerance)

    
    def get_profile_hmm_probs(
            self,
            msa_input,
            alphabet,
            alignment_type,
            global_exit_prob = None,
            match_threshold = 0.5,
            pseudocount = 1,
            gap = "-"
        ):
        # placeholder so the namespace exists, will replace with equivalent function from Linh's copy of the notebook once he's done editing it
        # pretend this returns trans_probs, emit_probs
        pass

    def _make_states_list(self, model_length: int) -> list[str]:
        """
        Function to generate a list of the states for a profile HMM of
        the given length

        Parameters
        ----------
        model_length : int
            number of match positions in the multiple sequence alignment used
            to make this profile HMM

        Returns
        -------
        list[str]
            List of the states, includes M0 
            and M{i}, I{i-1}, and D{i} for all i from 1 to model_length
        """
        # make a list of M{i} for all i from 0 to model_length
        matches = [f"M{i}" for i in range(model_length + 1)]

        # ditto for D{i} but starting at 1
        deletions = [f"D{i}" for i in range(1, model_length + 1)]

        # ditto for I{i-1}, again starting at 1
        insertions = [f"I{i-1}" for i in range(1, model_length + 1)]

        return matches + deletions + insertions
    
    def _trans_probs_to_dict(self, trans_probs: np.ndarray) -> dict[str, dict[str, float]]:
        """
        Function to convert transition probabilities matrix 
        from Numpy 3darray to dictionaries

        Parameters
        ----------
        trans_probs : np.ndarray
            3d numpy array where:
                Outer array indices represents match positions
                Middle array is state we're coming from, indexed matching ordered list of match, insert, delete
                Inner array is state we're going to, indexed matching above
            and Values in inner array are trans_prob(middle_array_state --> inner_array_state)

        Returns
        -------
        dict[str, dict[str, float]]
            dict mapping source state to a dict that, itself, maps destination state to
            the probability P(source -> destination)
        """
        # initialize output
        trans_dict = {}

        # iterate over the profile positions
        for i, array_i in enumerate(trans_probs):

            for j, row in enumerate(array_i):
                # make the key for the source state
                source_state = self.namelist[j] + str(i)

                # initialize as a defaultdict here so we correctly report 0 probability for illegal transitions (e.g. D4 to M0)
                trans_dict[source_state] = defaultdict(float)

                # now transition probabilities are listed in the current row, in order of M{i+1} at index 0
                trans_dict[source_state][f"M{i+1}"] = row[0]

                # I{i} is always at index 1
                trans_dict[source_state][f"I{i}"] = row[1]

                # and D{i+1} is always at index 2
                trans_dict[source_state][f"D{i+1}"] = row[2]
        
        return trans_dict

    def _emit_probs_to_dict(self, emit_probs: np.ndarray, alphabet: list[str], gap: str = "-") -> dict[str, dict[str, float]]:
        """
        Function to convert the emission probability matrix from a numpy 3darray
        to a dict of dicts

        Parameters
        ----------
        emit_probs : np.ndarray
            numpy 3darray for emission probabilities, where
                Index for outer array represents match position, i
			    Index for middle array represents whether you're looking at match states M{i} or insertion states I{i}
			    Index for inner array represents which emission
        alphabet: list[str]
            List of legal emissions
        gap: str, optional
            The character used to represent a gap in an aligned sequence, set as
            the guaranteed emission of the Deletion states, D{i} and has a probability of 0
            in insertion and match states
            "-" by default

        Returns
        -------
        dict[str, dict[str, float]]
            dict mapping state to a dict that, itself, maps emission name to P(emission | State)
        """
        # initialize our output
        emit_dict = {}

        # get a dict for the background emission probabilities to reuse for our insertion states
        # at any position, i, we can use index 1 for our middle array and it always gives us the same inner array
        background_array = emit_probs[0,1]
        background_dict = {alphabet[a]: background_array[a] for a in range(len(alphabet)) if alphabet[a] != gap}
        background_dict[gap] = 0.0

        # set up the same thing for the deletion states, except those aren't on emit_prob and have 
        # a probability of 0 for every emission
        deletion_dict = {emission: 0.0 for emission in alphabet}
        deletion_dict[gap] = 1.0

        # iterate over match positions
        for i in range(len(emit_probs)):
            
            # emit_probs[i][0] is always the match state array for this position
            # unpack the match state array into a dict
            match_dict = {alphabet[a]: emit_probs[i, 0, a] for a in range(len(alphabet))}
            match_dict[gap] = 0.0

            # now emit_probs[i][1] is always the insertion state array, which is always the same
            # so the insertion state array doesn't need to be unpacked again

            # now get all the states' entries written up
            emit_dict[f"M{i}"] = match_dict
            emit_dict[f"D{i}"] = deletion_dict
            emit_dict[f"I{i}"] = background_dict

        return emit_dict
