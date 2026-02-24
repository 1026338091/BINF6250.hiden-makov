# BINF6250 Project 4: Sequence Assembly

## Description of the project

This repository contains work done as part of the BINF6250 Algorithmic Foundations in Bioinformatics course (Northeastern University, Spring 2026). The goal is to implement a de Bruijn graph algorithm for genome assembly.

## Dependencies

## Contents

## Usage

## Pseudocode

```
Add to initialization code: 
Two dictionaries:
    The first maps a k-1mer to the number of times it appears as a prefix
    { k-1mer: #times_I'm_a_prefix }
    The second maps a k-1mer to the number of times it appears as a suffix
    { k-1mer: #times_I'm_a_suffix }
This lets us calculate balance more easily later!


The basic structure for our graph:
Dict mapping strings to a list:
Key: a prefix
Value: list of suffixes associated that


Function: Build graph from reads
Takes as input: 
    a list of reads
    a value for k (length of kmers)

Step 1: kmerize the reads (helper function this)
    Note: during this step, if the read is shorter than k, we just ignore it (if you want to use a high k that kicks out half of your data and you end up with a lot of small contigs/generally poor looking output, that's your perogative)
    Print out a count here of how many reads we threw out/ignored (print % ignored)

Step 2: For every kmer, do

    Step 3: get the prefix and suffix k-1mers

    Step 4: Add 1 to the prefix k-1mer's entry in the dict that counts how many times a given k-1mer appears as a prefix
            Ditto for the suffix, but using the dict that counts # of times it's a suffix

    Step 5: append the current suffix to the list associated with the current prefix (This is the add_edge helper function)
    NOTE: this adds duplicate list entries if a kmer is repeated (can't distinguish between kmers that repeat because reads overlap and kmers that repeat because the sequence appears multiple times in the genome)


Function: identify starts
Takes as input: 
    the graph
    prefix counts dict
    suffix counts dict
Returns as output:
    A list of prefixes that are start positions in our graph

Step 1: iterate through the prefixes, for each one do

    Step 2: just access the values for that prefix from both the prefix and suffix count dictionaries
            the value from the prefix counts dict is the number of edges that come out of it (because this is the number of times it's a prefix)
            the value from the suffix counts dict is the number of edges that come into it (because this is the number of times it's a suffix)

    Step 3: Subtract the number of edges out from the number of edges in (aka calculate in - out)
    
    Step 4: Check If that difference is 1
        If no, we have a balanced node or a semi-balanced node with more edges going into it (this would be an end point and we don't care about those right now)
        If yes, we have a semi-balanced node with more edges coming out of it than going into it- this is a start

    Step 5: If we have a semi-balanced node, we throw this prefix into the list of start positions

Step 6: after looping through prefixes, return the list of start positions


Function: remove edge: 
Takes as input: 
    Prefix k-1mer
    Suffix k-1mer

Step 1: Remove the suffix k-1mer from the list associated with the prefix k-1mer in the De Bruijn graph


Function: Eulerian Walk
Takes as input:
    Current position
    The graph (you can still access this via self.graph anyway)
    an RNG seed for reproducibility

Need to make sure that the node we're about to go to has edges coming out of it, and if it doesn't and we have other edges to go along, then we go along those other ones

Step 1: Randomly select a start position from the start positions list

Enter a loop of some fashion (while we still have edges to traverse?)

    Check if any of the suffixes that can come from the current position have edges that come out of them
        Make a list of the suffixes that do

    If we have suffixes to go to that still have edges to traverse, do the following:

        Step 2: Randomly select a suffix that can come from the current prefix

        Step 3: Remove the edge (helper function)

        Step 4: call this function, telling it that the suffix is it's "current position"

If our current position (current prefix) has no suffixes coming from it, we break from the loop and append the last letter of our current position to the path, which is a string
    NOTE: the jupyter notebook calls this the "tour", we call it the "path"



Function: assemble contigs
Takes as input:
    List of start positions (prefixes that are valid start positions)
    RNG seed (for reproducibility during testing)

Step 0: shuffle the order of the start positions list for fun (to get a differnt output)
    This avoids biasing the contigs based on whatever starting k-1mer appears first in the input fastq file

step 1: For each prefix in the start positions list, do

    step 2: Eulerian walk function

    step 3: grab the path variable and reverse it
        Note: The path variable is the concatenation of the last nucleotide in each k-1mer, added to the string in reverse order because of the recursive nature of the Eulerian walk
        2nd note: it will be missing all of the nucleotides from the starting k-1mer except for the last one
        3rd note: the fact that we tracked our path by concatenating the last nucleotide in each k-1mer + steps 3 and 4 that we're doing right now will invalidate the need for the tour_to_sequence() function
    
    step 4: concatenate all but the last character of the start position to the newly-reversed path string
        This is our contig that comes from this start <3
    
    Step 5: Add this contig to the list of contigs 


function get assembly stats:
takes list of contigs as input:

Number of contigs is just length of the list of contigs

Iterate over all the contigs
    Add their lengths to a list of lengths

Total length: Report the sum of the values in the lengths list

longest_contig: report the max from the list of contig lengths
shortest_contig: report the min from the list of contig lengths
mean_length: sum the values in the contig length list, divide by the length of the list

n50: N50 statistic (the shortest contig such that all contigs of at least that length cover half of the total length of all the contigs put together)
Make this a helper function that takes the contigs length list as an input:
    Start a value, n, at 0
    Sort the lengths list from largest value to smallest
    Iterate over the values in the newly-sorted lengths list
        Add the current value to n
        Check if n is greater than or equal to half of the total length (total length calculated above)
            If yes, return the length of the read
            If no, continue

Bundle all of these values we need to report into a tuple and report it


Function: write fasta
Takes as input: 
    list of contigs
    Desired filename

Step 1: open the output file

Step 2: iterate over the contigs (use enumerate so we access the index and the contig at the same time)

    Step 3: Print a header + the contig
        f-string with "> Contig {number} - newline character - Contig sequence"


```

## Personal reflections
### Successes

### Strugges

### Individual reflections

Justin Wildman:

Tien Nguyen:

Linh Nguyen (group leader):

## Appendix
