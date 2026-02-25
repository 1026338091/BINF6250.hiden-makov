# BINF6250 Project 4: Sequence Assembly

## Description of the project

This repository contains work completed for BINF6250 – Algorithmic Foundations in Bioinformatics (Northeastern University, Spring 2026).

The goal of this project is to implement a De Bruijn graph–based genome assembly algorithm from scratch in Python. The project reconstructs DNA contigs from short sequencing reads by:

1. Constructing a directed De Bruijn graph from k-mers

2. Tracking in-degree and out-degree of nodes

3. Identifying valid Eulerian path start positions based on each node's balance

4. Performing a non-recursive, stack-based Eulerian traversal

5. Assembling contigs from graph walks

6. Calculating standard assembly statistics (including N50)

This implementation emphasizes algorithmic clarity, correctness, and reproducibility.

## Dependencies
Python 3.9+

Standard Python libraries only:

    collections

    random

    time

    datetime

## Contents

`project04_mouse.ipynb`: The Jupyter notebook containing the final verson of our code for assembling the 10 million simulated reads from the mouse *mus musculus* genome. 

### Previous Builds and Testing Files:

`notebook_that_crashes_unless_you_export_it_to_python_script.ipynb`: A non-final version of our Jupyter notebook that would crash due to the inclusion of a flush print statement that we used in the python scipt version of our program to print our graph's build progress to the terminal. This was placed in an if statement in the final build so that we could avoid running it. 

`class_and_assembly_test.py`: This is the python script in which we did the bulk of our scripting for this project before porting the code to our Jupyter notebook. This script is actually runnable from the command line and generates most of the same outputs as the Jupyter file (with the major deviation being that this script has a simpler driver function that prints out fewer statistics). 

`python_script_that_does_not_crash.py`: This is a copy of `class_and_assembly_test.py` Linh used to make edits necessary for porting over our code from `class_and_assembly_test.py` to our Jupyter notebook. 

`project04_justin_copy.ipynb`: This is a copy of `project04_mouse.ipynb` Justin ported code into (from `class_and_assembly_test.py`) and edited for compatibility. The purpose of this was to stage changes for the code to be compatible with the jupyter notebook without altering our original script and without pushing changes to our main notebook before we were confident in them. 

## Usage

Make sure to download the gzipped simulated mouse fastq file, `mouse_SE_150bp.fq.gz` from the assignment page and place it in the same directory as the Jupyter notebook: `project04_mouse.ipynb`. This data is not stored on this repo due to its size and GitHub not being a data storage service. Aside from this prerequisite step, the script can be run directly off of the Jupyter notebook. As with any Jupyter notebook, the code cells must be run in order of apperaance. 

## Pseudocode

```
Add to initialization code: 
A balance dictionary:
    Maps a k-1mer to the balance of its node:
    We calculate this as (# times this k-1mer appears as a prefix) - (# times this k-1mer appears as a suffix)
    This, in effect, is (# edges out) - (# edges in)
    So when we look for start sites, we can look for nodes with a positive balance


The basic structure for our graph:
Dict mapping strings to a list:
Key: a prefix
Value: list of suffixes associated that


Function: Build graph from reads
Takes as input: 
    a list of reads
    a value for k (length of kmers)

Step 1: kmerize the reads
    Note: during this step, if the read is shorter than k, we just ignore it (if you want to use a high k that kicks out half of your data and you end up with a lot of small contigs/generally poor looking output, that's your perogative)
    Print out a count here of how many reads we threw out/ignored (print % ignored)
        We also chose to ignore reads with ambiguous bases as well. The breakdown of ignored reads and why they were ignored gets printed if we tell the function to be verbose

Step 2: For every kmer, do

    Step 3: get the prefix and suffix k-1mers

    Step 4: Add 1 to the prefix k-1mer's entry in balance dict
            Subtract 1 from the suffix k-1mer's entry in the balance dict

    Step 5: append the current suffix to the list associated with the current prefix (This is the add_edge helper function)
    NOTE: this adds duplicate list entries if a kmer is repeated (can't distinguish between kmers that repeat because reads overlap and kmers that repeat because the sequence appears multiple times in the genome)


Function: identify starts
Takes as input: 
    the graph
    balance dict
Returns as output:
    A list of prefixes that are start positions in our graph

Step 1: iterate through the prefixes, for each one do

    Step 2: just access the values for that prefix from the balance dict
    
    Step 3: Check If the prefix's balance is positive
        If no, we have a balanced node or a node with more edges going into it (this would be an end point and we don't care about those right now)
        If yes, we have a node with more edges coming out of it than going into it- this is a start

    Step 4: throw this prefix into the list of start positions

Step 5: after looping through prefixes, return the list of start positions


Function: remove edge: 
Takes as input: 
    Prefix k-1mer
    Suffix k-1mer

Step 1: Remove the suffix k-1mer from the list associated with the prefix k-1mer in the De Bruijn graph


Function: Eulerian Walk
Takes as input:
    A start node
    The graph (A deep copy of the graph from which we can remove edges without worrying about destroying/altering the original)

Need to make sure that the node we're about to go to has edges coming out of it, and if it doesn't and we have other edges to go along, then we go along those other ones

Initialize our stack by placing the start node we're using into it
    This stack will have nodes added in the order they're traversed in and unpacked in reverse order
Also initialize our path as an empty list
    Our path will have the nodes added in the reverse of whatever order they were traversed in
    NOTE: the jupyter notebook calls this the "tour", we call it the "path"

Enter a while loop that continues running until our stack is empty

    Update a value tracking the current node (our prefix) by setting it to whatever the last node in the stack is

    Check if the current node has any edges coming out of it

    If we have edges to traverse from our current position, do the following:

        Select the last suffix that can come from the current prefix (this is randomized because we shuffled every node's respective list of edges before calling this function)

        Remove the edge (helper function or just pop it)

        Add the selected suffix to the top of the stack

    Now if we don't have edges to traverse from our current position, we instead do this:

        Remove whatever node is on the top of the stack and put it into the path
            (We can just pop off the last node in the stack and append it to the path)

After the loop has finished, we now have an empty stack and a path with some number of nodes in it

Reverse our path so they will appear in the order we traversed them in (because this list of nodes had them added in reverse order)

Return the path (list of nodes)



Function: assemble contigs
Takes as input:
    our graph (has a start positions list saved as an attribute)
    RNG seed (for reproducibility during testing)

Step 0: shuffle the order of the start positions list (to get a differnt output) AND shuffle the order of the list of edges associated with each node
    This avoids biasing the contigs based on whatever starting k-1mer appears first in the input fastq file

step 1: For each prefix in the graph's start positions list, do

    step 2: Eulerian walk function
    
    step 3: concatenate the first k-1mer in the path and the last character of all the subsequent k-1mers
        This is our contig that comes from this path <3
    
    Step 4: Add this contig to the list of contigs

Step 5: Create contigs using all of the leftover k-1mers as start positions

Step 6: calculate all the contigs' lengths

Step 7: Calculate some quick diagnostic stats while we have the list of contig lengths already calculated
    This will include the stats we'll want to retrieve in get_assembly_stats() to avoid having to calculate contig lengths multiple times
    The diagnostic stats get saved to an attribute of the graph object


function get assembly stats:
takes no input because we already calculate these during assemble_contigs and save them to the graph object's 'stats' attribute


Report the following values we calculated earlier and saved to graph.stats:

    Number of contigs is just length of the list of contigs

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


Function: write fasta
Takes as input: 
    list of contigs
    Desired filename

Step 1: open the output file

Step 2: iterate over the contigs (use enumerate so we access the index and the contig at the same time)

    Step 3: Print a header + the contig + the length of the contig + kmer size
        f-string with "> Contig {number} - newline character - Contig sequence"


```

## Personal reflections
### Successes

Once we broke through how difficult it was to visualize the graph structure and how it relates to sequences of DNA, we had a very productive planning and pseudocoding meeting and were able to split up the scripting because we had so comprehensively planned out how we'd impelment everything that we all understood what all our code did. This worked out serendipitously because our schedules over the weekend didn't line up well for trying to meet, and once we met up, we only had a few small optimizations and tweaks to catch up on before we were able to work together on putting our code into the notebook and doing further debugging. 

We found a lot of success in figuring out ways in which we could optimize the memory load of our graph and how we handle all of the reads. Of particular note is that we were able to modify the `read_fastq` function to be a generator, rather than returning a list of reads, as we did not need to store all 10 million reads (approx 1.5 billion bp) in memory all at once because we only wanted to work with them  one at a time. 

### Strugges
Understanding why Eulerian paths must append nodes in reverse order.

Debugging recursive traversal and ensuring edges were removed correctly.

Managing graph copies to avoid modifying the original structure.

Handling performance challenges with large datasets.

We had a lot of difficulty during our initial planning meeting as we tried to break down the how the De Bruijn graph should be expected to work. It was particularly difficult at first to communicate our ideas because it was difficult for us to visualize our graph and how it related to the reads that the kmers and k-1mers came from. We actually opened up drawing apps and shared our screens so we could help each other figure out how our graph will look and how it relates to the DNA sequences it was made with and will assemble. We broke through the initial difficulties with visualizing the graph, traversal, and how that all relates to DNA sequences, particularly thanks to grabbing actual example graphs and walking through the traversal by hand, which let us realize that the graphs will only branch if they're also circular. Once we were able to visualize everything, the problem got a lot easier for us to solve and we had a very successful and productive pseudocoding/planning session. 

We found out the hard way just how important optimization is, as the memory footprint of a graph based on the 10 million read dataset was so large that, if unoptimized, our computers were flat-out unable to run our scripts. Additionally, we ran into recursion errors while testing our graphs on the larger dataset, which led to us opting to use a stack appraoch. Thanks to the stack being first-in last-out, it has the same effect of adding the nodes to our tour in reverse order as recursion would, without the problem with hitting the recursion limit. 

### Individual reflections

Justin Wildman: While I feel like I could have contributed more to the scripting, I feel happy with how involved I was with the pseudocode/planning and the process of moving our code into the jupyter notebook. We had some difficulty with visualizing our graph and how we'd handle any kind of branching. Once we got all of our uncertainties hashed out, we had an amazingly productive evening planning everything out, and we even planned everything out to the degree that we were all intimately aware of how each part of our code would be implemented and could script pieces of it on our own time. This was quite fortunate because I was busy over the weekend and unable to touch the project for a couple days. I'm really thankful Linh and Tien were understanding and were able to cover a good portion of the scripting, and they were super helpful in getting me up to speed with the relatively few tweaks and optimizations they'd found necessary. Debugging everything and moving it all into the notebook ended up being quite the laborious process, and I'm happy I got to contribute there too. The driver function made a lot of assumptions about the nature of the graph and the contigs that conflicted with how we implemented our graph (particularly that we would save the list of contigs as a list of strings outside of the graph object and then pass it to the class methods, where we instead just made our class methods access and work from the graph's attribute that saved the list of nodes in each path. It was a lot of work for us to remove all the landmines that came from how the notebook we were given would load all the reads into memmory (SUPER unnecessary!) and how the driver function assumed we'd implemented everything. It was very satisfying to bring it all together, though truthfully after I'd finished helping bring all of our code from our python script into the notebook, I wasn't 100% sure it was fine because my computer wouldn't allocate enough RAM (maybe if I closed my chrome tabs, but I need those tabs open for my other classes or I WILL forget assignments :sob:) to actually run the script. Luckily, Linh's machine had a lot of extra RAM to work with, and he confirmed that everything was in working order. Working with Tien and Linh was a very enriching experience, as Linh was a lot more experienced with computation and, especially, optimizing everything than I was, so I got to learn a lot from my team about handling the memory concerns we have to navigate with such large graphs. 

Tien Nguyen: From this assignment, I learned how the De Bruijn graph algorithm and Eulerian walk are applied in genome assembly and how theoretical concepts are translated into practical code. I was able to successfully implement and test the algorithm using toy examples, which helped me understand how k-mers, nodes, and edges interact within the graph. However, I found it challenging to scale the script to run efficiently on real sequencing data due to memory and performance limitations, which required me to think more carefully about optimization strategies such as using iterators and reducing unnecessary data storage. Working with Linh and Justin was a positive experience, as we collaborated closely to develop the pseudocode during our first meeting and supported each other throughout debugging and testing. This project strengthened my problem-solving skills, improved my understanding of recursive algorithms, and increased my confidence in working with large biological datasets.

Linh Nguyen (group leader): The first meeting when we collaborated to grasp the central algorithm was very satisfying; we initially overcomplicated a lot before we understood how the simple recursive logic worked by visually going through realistic graphs together. The excitement about the mathematical elegance of a 5-line recursive algorithm was then dampened by the reality that it's almost never practical to use on non-toy datasets. We tried our best to match the structure of the driver function's printout after making performance optimizing changes to the workflow, but we're pretty sure that if we coded the way the original assignment notebook seems to imply we should, the script would not run nearly as well (or at all) on our computers. Having to optimize the code also served as a demonstration(/reminder) of those introductory Python buzzphrases like "strings are immutable", and got us thinking about what constitutes redundant information. It also allowed us to get some practice using non-toy iterators. Also, at first we for some reason just thought of reasonable numbers for k as being rather low, which adds some artificial computational AND interpretation challenges. Looking back, that was evidence of not actually fully understanding why the algorithm works, so I'm glad we finally got to setting k to 150 for a dataset of only 150-length perfect reads.

## Appendix
https://www.geeksforgeeks.org/dsa/hierholzers-algorithm-directed-graph/ we didn't make up stack methods.
