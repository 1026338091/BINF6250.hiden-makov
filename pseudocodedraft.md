note: we can make object referrable by "either index or name" in a lot of places, i'll call that "ref". so if an emission_set named "xyz" is the first one in a model's schema, its "ref" when looking it up in the emission_set_list is either the string xyz or the index 0.

# class emission_set
represents one named set of mutually exclusive possible observed values.

### attributes:
- set_name: optional but recommended identifier for the emission set. if used, should be unique within an `HMM`.
- length: number of emission values in the set. always equal to the number of value names.
- value_names: list[str] of strings naming the emission values in this set.
- default_weights: list[float] of of numeric weights, one per emission value. used as fallback weights when a hidden state does not yet have a specific weight vector for this emission set.

### methods:
- `initialize(name: str | None = None, length: int, value_names: list[str] | None = None, default_weights: list[float] | None = None)`: 
	- if `value_names` is omitted, initialize as unnamed values of the given length. If `default_weights` is omitted, use zeros.
- replace set_name
- replace full value_names list
- replace full default_weights vector
- replace value name at index
- replace default_weight at index 
- `add_emission_value(value_name: str, default_weight`: float = 0): add emission value to the set (append), incrementing length by 1
- ...

---------

# class hidden_state
represents one hidden state in the HMM, including its init weight and its state-specific emission weights. 

### attributes:
- hidden_state_name: str | none
- init_weight: default init weight, if not set default to 0
- `emission_weights: dict[str, list[float]]` : dictionary of lists of weights that map emission set names to weights

### methods
- initialize(name: str | None = None, init_weight: float = 0, emission_weights: dict[str, list[float]] | None = None):
- replace hidden_state_name
- replace init_weight
- replace the full weight vector for one emission set at ref
- replace one emission weight within one emission set (using two refs)
- ...

-------

# class HMM
- stores the full hidden Markov model
- keeps emission sets and hidden states consistent
- stores transition weights
- derives normalized probabilities when requested.

### attributes
- `emission_set_list`: list of emission_set objects for the model. this defines allowed emission schema.
- `hidden_states_list`: list of hidden_state objects for the model.
- `W_hh`: transition weight matrix ie `W_hh[pref state ref][current state ref]`; always updated when there is a change in hidden_states_list
- `W_eh`: nested dictionary representation of emission weights ie `W_eh[emission set ref][hidden state ref][emission value ref] = a weight`. Derived from the hidden states and emission sets only when requested
- `P_hh, P_eh, P_init`: Normalized transition/emission/init probability matrix/dict/dict. Created only when requested, and only "saved" when `normalize_all()`.

### methods

#### lookup methods
- get_emission_set(emission_set_ref), returns emission set object 
- get_hidden_state(hidden_state_ref): same idea but for hs
- get_emission_set_index(emission_set_name) -> int
- get_hidden_set_index(hidden_state_name) -> int
- ...

#### model building methods

`initialize(emission_sets: list[emission_set] | None = None, hidden_states: list[hidden_state] | None = None)`:
	- start with empty `emission_sets`, `hidden_states`, and `W_hh`
	- add provided emission sets
	- add provided hidden states
	- size `W_hh` to match the number of hidden states; set transition weights to `0`


`add_emission_set(new_emission_set, fill_hidden_states_with="zeros" or "default")`:
	- require the emission set name to be new within the model
	- append the emission set to `emission_sets`
	- for each existing hidden state that lacks this emission set:
	    - add a weight vector of zeros, or
	    - add the emission set’s `default_weights`

`add_hidden_state()` :
- arguments:
	- `new_hidden_state: hidden_state`
	- `mode: str`  = `"strict"` or `"force"`
	- `missing_fill: str`  
	used only in `"force"` mode. `"zeros"` or `"default"`
	- `incoming_transition_weights: list[float] | dict[str, float] | None`  
		   weights for transitions into the state.
	- `outgoing_transition_weights: list[float] | dict[str, float] | None`  
		    weights for transitions out of the state.
	- `self_transition_weight: float`  
		    weight for transition from the new state to itself.
	- `update_init_weight: float | None`  
		    if given, replaces the state’s `init_weight`.
- in all modes:
	- require the new state name to be unique
	- append the state to `hidden_states`
	- enlarge `W_hh` by one row and one column
	- set all unspecified new transition weights to `0`
	- if `update_init_weight` is provided, replace the state’s initial weight
- in `strict` mode: 
	- (we can start development with only strict mode, it's easier)
	- the new state must contain exactly the model’s emission set names
	- no emission set may be missing
	- no extra emission set may be present
	- each emission weight vector must match the corresponding emission set length
- in `force` mode:
	- for each model emission set missing from the new state:
	    - insert a weight vector filled with zeros, or
	    - insert that emission set’s `default_weights`
	- reject any emission set name not already present in the model
- transition matrix update:
	- expand `W_hh`
	- insert incoming transition weights
	- insert outgoing transition weights
	- set the self-transition (last row last column)
	- leave unspecified transitions at 0 (or whatever)

`fill_missing_state_emissions(hidden_state_ref, method="zeros")`: 
	- for any emission set present in the model but missing from the state, insert a matching weight vector. 
	- used mainly during forced add_hidden_state

other methods:
	- replace hidden state by ref
	- replace transition weight by ref x ref
	- replace transition row for one "prev state" by ref
	- replace transition column for one "current state" by ref
	- replace init weight by ref
	- ...

#### explicitly describing a design choice
the model’s emission schema is defined by `emission_sets`, and all hidden states must conform to it. strict mode requires exact agreement. force mode fills missing emission weights; so it allows for adding models which LACK weights for certain emission sets already in the schema. extra emission sets are always rejected unless added to the model first via `add_emission_set`.

#### matrix/dict building methods
- `build_W_eh()`
- `build_P_init()`
- `build_P_hh()`
- `build_P_eh()`
- `normalize_all()`: calls the three above and stores in object rather than return

- validation & printing methods....
-...

------

# viterbi(emission, HMModel)

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
```

# truly recursive viterbi (don't use this)

```
n = len(emissions)

states = [all possible hidden states in index order] 

scores = (len(states) x n) matrix of -inf

traceback = (len(states) x n) matrix of nones

def recurfun(i, curr_s, prev_s):
	
	best_score = scores[i][curr_s]
	best_prev = traceback[i][curr_s]
	
	if best_score != -inf:
		return
	if i == 0:
		scores[0][s] = init probability of s * P(emission[0]|s) 
	
	emit_p = P(i|curr_s)
	
	for h in prev_s:

		prior = recurfun(prev_s)
		trans_p = P(prev_s -> curr_s)
		score = prior * trans_p * emit_p
		
		if score > best_score:
			best_score = score
			best_prev = prev_s	

best_final_score = -inf

for s in states:
	score = recurfun(n-1, s)
	
	if score > best_final_score:
		best_final_score = score
		reverse_path = [s]

for i in emission[n] to emmission[0]:
	
	state_at_i = reverse_path[-1]
	where_from = traceback[i][state_at_i]
	reversed_path.append(where_from)

```

# train

start with: text files

x=
seq_bases
seq_methylation

y=
seq_hidden_states

parse

make them into lists

for all positions:
- prev "state" depends on order
- current state is current in seq_hidden_states <- either this or do only first order and use products
- you have to do the beta thing

index using "current base" position

goal:
- transition weights -> dict{prev:{current:weight}}
- emission weights -> dict{state:{emission:weight}}
- initial weights -> dict{state:weight}

update/output:
- list of EmissionSet objects (identities)
- list of HiddenState objects with preloaded emission weights and initial weights
- transition weight dict to load columns/rows one at a time into W_hh HMModel

