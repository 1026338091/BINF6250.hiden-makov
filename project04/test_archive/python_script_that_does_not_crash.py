# %% [markdown]
# # De Bruijn graphs

# %%
from collections import defaultdict
import random
import gzip

# %%
# external helpers
def _eulerian_walk(graph_to_consume, startnode):
    stack = [startnode]
    path = []

    while stack:
        currnode = stack[-1]
        if graph_to_consume[currnode]:
            # go to the next node along one of the edges
            nextnode = graph_to_consume[currnode].pop()
            stack.append(nextnode)
        else:
            # if there are no edges to go to from the current node, start emptying the stack into the path
            path.append(stack.pop())

    # reverse the list of nodes we've been to because they're added in reverse order (last-in, first-out)
    path.reverse()
    return path

def _get_n50(contig_lengths: list[int]) -> int:
    """
    Take a list of contigs' lengths, sorted from greatest to least, and calculate the shortest contig for which
    all contigs of that length or longer, when put together, represent 50% or more of the assembly

    @param contig_lengths: list of ints representing the lengths of contigs, MUST be sorted from highest to lowest
    @return: int, the n50 statistic
    """
    total_length = sum(contig_lengths)

    # initialize a value to track the total length covered by the reads we're about to iterate over
    dist = 0

    for curr_length in contig_lengths:
        # add the current contig's length to our sum
        dist += curr_length
        # check if our sum is >= half of the total length of the contigs
        if dist >= total_length / 2:
            # quit out because we've hit the shortest contig that meets our conditions
            return curr_length

def tour_to_sequence(nodes):
    # list of (k-1)-mers -> contig string
    return nodes[0] + "".join(n[-1] for n in nodes[1:])

# %%
def iter_fastq_seqs_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        i=0
         
        while True:
            i +=1
            header = fh.readline()
        
            if not header:
                return  # EOF
            # if i > 100000000:
            #     return 

            print(f'\r[build] progress: {i} reads', end='', flush=True)

            seq = fh.readline()
            plus = fh.readline()
            qual = fh.readline()

            yield seq.strip()

# %%
class DeBruijnGraph:

    def __init__(self, reads, k, 
    ignore_ambiguous=True, verbose=True): # mostly for debugging
        self.k = k
        self.ignore_ambiguous = ignore_ambiguous
        self.graph = defaultdict(list)
        self.balance = defaultdict(int)
        self.startnodes = []
        self.stats = {}
        self.contig_lists = None  # list of node-path lists created by .assemble_contigs()
        self.build_graph_from_reads(reads, verbose=verbose)

    def build_graph_from_reads(self, reads, verbose=True):
        reads_total = 0
        ignored_short = 0
        ignored_amb = 0
        bases_total = 0
        kmers_added = 0
        avg_read_length = 0


        for read in reads:
            read = read.strip().upper()
            reads_total += 1
            bases_total += 1
            avg_read_length = (bases_total + len(read))/reads_total

            if len(read) < self.k:
                ignored_short += 1
                continue # escapes read's loop, ignoring it because it's too short

            if self.ignore_ambiguous and any(b not in "ACGT" for b in read):
                ignored_amb += 1
                continue # escapes read's loop, ignoring it because there are ambiguous nucleotides in the read

            for i in range(len(read) - self.k + 1):
                # loops through kmers of read
                kmer = read[i : i + self.k]

                prefix = kmer[:-1]
                suffix = kmer[1:]

                # add an entry into the graph indicating that the current suffix's node  is connected to the current prefix's node via an edge
                self.graph[prefix].append(suffix)

                self.balance[prefix] += 1 # indicates another edge leaves this prefix's node
                self.balance[suffix] -= 1 # indicates another edge enters this prefix's node

                kmers_added += 1

        # make a list of nodes (k-1mers) for which they have more edges entering them than leaving
        self.startnodes = [n for (n, b) in self.balance.items() if b > 0]

        edges = sum(len(adj) for adj in self.graph.values())
        nodes = len(set(self.balance.keys()) | set(self.graph.keys()))
        bal_pos = sum(1 for b in self.balance.values() if b > 0) # number of nodes with more edges going out than coming in
        bal_neg = sum(1 for b in self.balance.values() if b < 0) # number of nodes with more edges coming in than going out
        bal_zero = sum(1 for b in self.balance.values() if b == 0) # number of balanced nodes
        bal_abs_gt1 = sum(1 for b in self.balance.values() if abs(b) > 1) # number of nodes unbalanced by more than 1 edge

        self.stats = {
            "reads_total": reads_total,
            "bases_total": bases_total,
            "avg_read_length": avg_read_length,
            "reads_ignored_too_short": ignored_short,
            "reads_ignored_ambiguous": ignored_amb,
            "kmers_added": kmers_added,
            "nodes": nodes,
            "edges": edges,
            "balance_pos": bal_pos,
            "balance_neg": bal_neg,
            "balance_zero": bal_zero,
            "balance_abs_gt1": bal_abs_gt1,
            "start_nodes": len(self.startnodes),
        }

        if verbose:
            ignored = ignored_short + ignored_amb
            pct_ignored = (ignored / reads_total * 100.0) if reads_total else 0.0
            print(f"[build] k={self.k}")
            print(f"[build] received {reads_total} reads; ignored {ignored} ({pct_ignored:.2f}%)")
            if ignored_short:
                print(f"[build] ignored because too short={ignored_short}")
            if ignored_amb:
                print(f"[build] ignored because ambiguous base(s)={ignored_amb}")
            print(f"[build] {nodes} nodes; {edges} edges ({kmers_added} kmers)")
            print(f"""[build] balance: +={bal_pos} -={bal_neg}; {bal_zero} 0s; |abs(b)|>1={bal_abs_gt1}""")
            if bal_abs_gt1:
                print("[build] warning: some nodes have absolute balance > 1")
            if not self.startnodes and edges:
                print("[build] warning: no balance > 0 start nodes detected; assembly will rely on leftover cleanup step.")

    def assemble_contigs(self, seed=None, verbose=True):
        # uses working copy of the graph
        # policy:
        # - prioritize balance > 0 start nodes (shuffled deterministically by seed)
        # - then use up any leftover nodes with outgoing edges until graph exhausted
        # - stores contigs as node paths in self.contig_lists

        rng = random.Random(seed) # local rng

        # make a deep copy of the graph so we can remove elements for our walk without editing the original graph object
        working_graph = defaultdict(list, {currentnode:nextnodes_list.copy() for currentnode, nextnodes_list in self.graph.items()})

        for currentnode in working_graph:
            rng.shuffle(working_graph[currentnode]) # shuffle edge lists

        starts = list(self.startnodes)
        rng.shuffle(starts) # shuffle starts

        contigs = []

        # prioritize balance-based start candidates
        for s in starts:
            path_nodes_list = _eulerian_walk(working_graph, s)
            contigs.append(path_nodes_list)

        # exhaust leftovers
        # there's gotta be a prettier way to do this
        while True:
            next_start = None

            for currentnode, nextnodes_list in working_graph.items():
                # are there any available currentnodes to make more contigs?
                if nextnodes_list: 
                    next_start = currentnode
                    break

            if next_start is None:
                # didn't find any
                break

            path_nodes_list = _eulerian_walk(working_graph, next_start) # make another contig
            contigs.append(path_nodes_list)

        self.contig_lists = contigs

        # assembly diagnostics
        # add actual stats after this is just for debugging the assembly itself
        leftover_edges = sum(len(adj) for adj in working_graph.values())
        contig_lengths = [0 if not p else (len(p) + self.k - 2) for p in contigs]
        contig_lengths.sort(reverse=True)

        self.stats.update({
            "assembly_seed": seed,
            "num_contigs": len(contigs),
            "total_length": sum(contig_lengths),
            "leftover_edges_after_exhaust": leftover_edges
        }) #qc stats

        self.stats.update({
            "longest_contig": contig_lengths[0] if contig_lengths else 0,
            "shortest_contig": contig_lengths[-1] if contig_lengths else 0,
            "mean_length": self.stats['total_length']/self.stats['num_contigs'],
            "n50": _get_n50(contig_lengths) if contig_lengths else 0
            })

        if verbose:
            print(f"[assembly] seed = {seed}; contigs = {self.stats['num_contigs']}")
            print(f"[assembly] leftover edges = {leftover_edges}")
            print()

        return contigs

    def get_assembly_stats(self):

        return {
        "num_contigs": self.stats["num_contigs"],
        "total_length": self.stats["total_length"],
        "longest_contig": self.stats["longest_contig"],
        "shortest_contig": self.stats["shortest_contig"],
        "mean_length": self.stats["mean_length"],
        "n50": self.stats["n50"],
        }


    def write_fasta(self, out_path, sort_by_len=True, wrap=60):

        if self.contig_lists is None:
            raise RuntimeError("assemble_contigs() has not been run yet.")
        contigs = []

        for i, path_nodes in enumerate(self.contig_lists, start=1):
            # convert the path from a list of nodes to the actual sequence it represents
            seq = tour_to_sequence(path_nodes)
            contigs.append((i, seq))

        if sort_by_len:
            contigs.sort(key=lambda x: len(x[1]), reverse=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for idx, seq in contigs:
                f.write(f">contig_{idx:06d} len={len(seq)} k={self.k}\n")
                for i in range(0, len(seq), wrap):
                    f.write(seq[i:i+wrap] + "\n")

# %% [markdown]
# ---
# # Toy Example

# %%
print("="*60)
print("EXAMPLE 1: Assembling 9 overlapping reads into one contig")
print("="*60)

toy_reads_1 = [
    "ATGGCGTACG",  # Read 1
    "GGCGTACGTT",  # Read 2: overlaps with Read 1
    "CGTACGTTAC",  # Read 3: overlaps with Read 2
    "TACGTTACCA",  # Read 4: overlaps with Read 3
    "CGTTACCATG",  # Read 5: overlaps with Read 4
    "TTACCATGGG",  # Read 6: overlaps with Read 5
    "ACCATGGGCC",  # Read 7: overlaps with Read 6
    "CATGGGCCTA",  # Read 8: overlaps with Read 7
    "TGGGCCTAAA"   # Read 9: overlaps with Read 8
]

print(f"\nInput: {len(toy_reads_1)} reads")
print("First read:  ", toy_reads_1[0])
print("Last read:   ", toy_reads_1[-1])
k=9
print(f"\nBuilding De Bruijn graph with k=6...")

# 1) Build the graph
dbg = DeBruijnGraph(toy_reads_1, k=k)

# Optional: quick sanity checks
num_nodes = len(dbg.graph)
num_edges = sum(len(v) for v in dbg.graph.values())
print(f"Graph has {num_nodes} nodes and {num_edges} edges")

# 2) Assemble contigs
print("\nAssembling contigs...")
contigs = dbg.assemble_contigs(seed=42)

print(f"\nAssembled {len(contigs)} contig(s):")
for i, c in enumerate(contigs, start=1):
    seq = tour_to_sequence(c)
    print(f"  Contig {i}: length={len(seq)}")
    print(f"    {seq}")

# 3) Stats
stats = dbg.get_assembly_stats()
print("\nAssembly stats:")
for k_stat, v in stats.items():
    print(f"  {k_stat}: {v}")

# 4) (Optional) write to fasta
dbg.write_fasta("test.fasta")
print("\nWrote FASTA to toy_example_1_contigs.fasta")

# %% [markdown]
# ---
# # Real data
# This section utilizes a FASTQ file with 10 million "perfect" 150 bp reads simulated
# from the mouse genome (`GRCm39`) and tasks your program to ingest, assemble, and traverse
# the genome with statistical output report.

# %%
"""Driver program for mouse genome assembly using De Bruijn graphs.

This script demonstrates how to use the completed DeBruijnGraph class to
assemble 10 million perfect 150bp single-end reads from the mouse genome.

Usage:
    Run all cells in order in a Jupyter notebook.

Expected input:
    - File: mouse_SE_150bp.fq
    - Format: FASTQ
    - Reads: 10 million perfect 150bp single-end reads
    - Source: Simulated from mouse genome using wgsim

Output:
    - mouse_assembly.fasta: Assembled contigs
    - mouse_assembly_stats.txt: Detailed statistics
"""

import time
from datetime import datetime


def write_statistics_file(
    stats_file,
    input_file,
    num_reads,
    avg_read_length,
    k_mer_size,
    random_seed,
    num_nodes,
    num_edges,
    stats,
    contig_lengths,
    timing,
    coverage_estimate,
    assembly_fraction
):
    """Write comprehensive assembly statistics to text file.
    
    Args:
        stats_file (str): Output filename.
        input_file (str): Input FASTQ filename.
        num_reads (int): Number of reads processed.
        avg_read_length (float): Average read length.
        k_mer_size (int): K-mer size used.
        random_seed (int): Random seed used.
        num_nodes (int): Number of graph nodes.
        num_edges (int): Number of graph edges.
        stats (dict): Assembly statistics.
        contig_lengths (list): Sorted list of contig lengths.
        timing (dict): Timing information.
        coverage_estimate (float): Estimated sequencing coverage.
        assembly_fraction (float): Assembly size as % of genome.
    """
    with open(stats_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MOUSE GENOME ASSEMBLY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File: {input_file}\n")
        f.write(f"K-mer Size: {k_mer_size}\n")
        f.write(f"Random Seed: {random_seed}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("INPUT DATA\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of reads:         {num_reads:,}\n")
        f.write(f"Average read length:     {avg_read_length:.1f} bp\n")
        f.write(f"Total sequencing data:   {num_reads * avg_read_length:,.0f} bp\n")
        f.write(f"Estimated coverage:      {coverage_estimate:.1f}x\n")
        #f.write(f"Read time:               {timing['read_time']:.2f} seconds\n\n")
        
        f.write("-"*80 + "\n")
        f.write("DE BRUIJN GRAPH CONSTRUCTION\n")
        f.write("-"*80 + "\n")
        f.write(f"Graph nodes:             {num_nodes:,}\n")
        f.write(f"Graph edges:             {num_edges:,}\n")
        f.write(f"Average out-degree:      {num_edges/num_nodes:.2f}\n")
        f.write(f"Construction time:       {timing['graph_time']:.2f} seconds\n\n")
        
        f.write("-"*80 + "\n")
        f.write("ASSEMBLY RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of contigs:       {stats['num_contigs']:,}\n")
        f.write(f"Total assembly length:   {stats['total_length']:,} bp\n")
        #f.write(f"Assembly vs. genome:     {assembly_fraction:.2f}%\n")
        f.write(f"Longest contig:          {stats['longest_contig']:,} bp\n")
        f.write(f"Shortest contig:         {stats['shortest_contig']:,} bp\n")
        f.write(f"Mean contig length:      {stats['mean_length']:,.1f} bp\n")
        f.write(f"N50:                     {stats['n50']:,} bp\n")
        f.write(f"Assembly time:           {timing['assembly_time']:.2f} seconds\n\n")
        
        f.write("-"*80 + "\n")
        f.write("TOP 20 LONGEST CONTIGS\n")
        f.write("-"*80 + "\n")
        for i, length in enumerate(contig_lengths[:20], 1):
            f.write(f"{i:3d}. {length:10,} bp\n")
        f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("CONTIG LENGTH DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        bins = [
            (">100kb", sum(1 for x in contig_lengths if x > 100000)),
            (">50kb", sum(1 for x in contig_lengths if x > 50000)),
            (">10kb", sum(1 for x in contig_lengths if x > 10000)),
            (">5kb", sum(1 for x in contig_lengths if x > 5000)),
            (">1kb", sum(1 for x in contig_lengths if x > 1000)),
            (">500bp", sum(1 for x in contig_lengths if x > 500)),
        ]
        for bin_name, count in bins:
            f.write(f"Contigs {bin_name:8s}:     {count:,}\n")
        f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("TIMING SUMMARY\n")
        f.write("-"*80 + "\n")
        total_time = timing['total_time']
        #f.write(f"Read time:               {timing['read_time']:8.2f} seconds "
        #        f"({timing['read_time']/total_time*100:5.1f}%)\n")
        f.write(f"Graph construction:      {timing['graph_time']:8.2f} seconds "
                f"({timing['graph_time']/total_time*100:5.1f}%)\n")
        f.write(f"Assembly:                {timing['assembly_time']:8.2f} seconds "
                f"({timing['assembly_time']/total_time*100:5.1f}%)\n")
        f.write(f"Total time:              {total_time:8.2f} seconds "
                f"({total_time/60:.2f} minutes)\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")


def assemble_mouse_genome(
    input_file="mouse_SE_150bp.fq.gz",
    output_fasta="mouse_assembly.fasta", 
    stats_file="mouse_assembly_stats.txt",
    k_mer_size = 150,
    random_seed = 12345
):
    """Main driver function for mouse genome assembly.
    
    This function orchestrates the complete assembly pipeline:
    1. Load FASTQ reads
    2. Build De Bruijn graph
    3. Assemble contigs
    4. Calculate statistics
    5. Write output files
    
    Returns:
        dict: Dictionary containing assembly results including:
            - dbg: DeBruijnGraph object
            - contigs: List of assembled sequences
            - stats: Assembly statistics dictionary
            - timing: Performance timing information
    """
    print("="*80)
    print("MOUSE GENOME ASSEMBLY PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    timing = {}
    timing["read_time"] = 0.0 # iterator
    t0 = time.time()

    print(f"Time elapsed: {timing['read_time']:.2f} seconds")
    
    print("STEP 1: Building De Bruijn graph")
    print("-"*80)
    print(f"K-mer size: {k_mer_size}")
    print()
    
    reads_iter = iter_fastq_seqs_gz(input_file)
    dbg = DeBruijnGraph(reads_iter, k=k_mer_size)
    timing["graph_time"] = time.time() - t0

    num_reads = dbg.stats["reads_total"]
    total_bases = dbg.stats["bases_total"]
    avg_read_length = dbg.stats["avg_read_length"]
    
    # Calculate graph statistics
    num_nodes = dbg.stats["nodes"]
    num_edges = dbg.stats["edges"]
    avg_degree = num_edges/num_nodes if num_nodes > 0 else 0
    
    print(f"Graph construction complete!")
    print(f"Reads loaded: {num_reads:,}")
    print(f"Total bases (input): {total_bases:,} bp")
    print(f"Average read length: {avg_read_length:.1f} bp")
    print(f"  Nodes (unique {k_mer_size-1}-mers): {num_nodes:,}")
    print(f"  Edges (k-mer transitions): {num_edges:,}")
    print(f"  Average out-degree: {avg_degree:.2f}")
    print(f"Time elapsed: {timing['graph_time']:.2f} seconds")
    print()
    
    print("STEP 2: Assembling contigs")
    print("-"*80)
    print(f"Finding connected components and traversing graph...")
    print(f"Random seed: {random_seed} (for reproducibility)")
    print()
    
    t0 = time.time()
    contigs = dbg.assemble_contigs(seed=random_seed)
    timing["assembly_time"] = time.time() - t0
    
    print(f"Assembly complete!")
    print(f"  Contigs generated: {dbg.stats['num_contigs'],}")
    print(f"Time elapsed: {timing['assembly_time']:.2f} seconds")
    print()
    
    print("STEP 3: Calculating assembly statistics")
    print("-"*80)
    
    stats = dbg.get_assembly_stats()
    
    print(f"Assembly Statistics:")
    print(f"  Number of contigs:     {stats['num_contigs']:,}")
    print(f"  Total assembly length: {stats['total_length']:,} bp")
    print(f"  Longest contig:        {stats['longest_contig']:,} bp")
    print(f"  Shortest contig:       {stats['shortest_contig']:,} bp")
    print(f"  Mean contig length:    {stats['mean_length']:,.1f} bp")
    print(f"  N50:                   {stats['n50']:,} bp")
    print()
    
    # Display distribution of contig lengths
    
    contig_lengths = [0 if not p else (len(p) + k_mer_size - 2) for p in dbg.contig_lists]
    contig_lengths.sort(reverse=True)

    print(f"Contig Length Distribution:")
    print(f"  Top 10 longest contigs:")
    for i, length in enumerate(contig_lengths[:10], 1):
        print(f"    {i:2d}. {length:,} bp")
    print()
    
    # Calculate coverage estimate
    genome_size_estimate = 2700000000  # Mouse genome ~2.7 Gbp
    coverage_estimate = (num_reads * avg_read_length) / genome_size_estimate
    assembly_fraction = (stats['total_length'] / genome_size_estimate) * 100
    
    print(f"Genome Coverage Analysis:")
    print(f"  Mouse genome size (expected): ~{genome_size_estimate:,} bp")
    print(f"  Estimated sequencing coverage: {coverage_estimate:.1f}x")
    print(f"  Assembly size vs. genome: {assembly_fraction:.1f}%")
    print()
    
    print("STEP 4: Writing output files")
    print("-"*80)
    
    # Write assembled contigs to FASTA
    dbg.write_fasta(output_fasta)
    print(f"✓ Contigs written to: {output_fasta}")
    
    timing["total_time"] = timing["graph_time"] + timing["assembly_time"]

    # Write detailed statistics
    write_statistics_file(
        stats_file,
        input_file,
        num_reads,
        avg_read_length,
        k_mer_size,
        random_seed,
        num_nodes,
        num_edges,
        stats,
        contig_lengths,
        timing,
        coverage_estimate,
        assembly_fraction
    )
    print(f"✓ Statistics written to: {stats_file}")
    print()
    
    timing['total_time'] = (timing['graph_time'] + timing['assembly_time'])
    
    print("="*80)
    print("ASSEMBLY COMPLETE")
    print("="*80)
    print(f"Total time: {timing['total_time']:.2f} seconds "
          f"({timing['total_time']/60:.2f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print(f"Summary:")
    print(f"  • Processed {num_reads:,} reads")
    print(f"  • Built graph with {num_nodes:,} nodes and {num_edges:,} edges")
    print(f"  • Assembled {stats['num_contigs']:,} contigs")
    print(f"  • Total assembly: {stats['total_length']:,} bp (N50: {stats['n50']:,} bp)")
    print(f"  • Output files: {output_fasta}, {stats_file}")
    print()
    
    return {
        'dbg': dbg,
        'contigs': dbg.contig_lists,
        'stats': stats,
        'timing': timing,
        'graph_stats': {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree
        },
        'coverage': coverage_estimate
    }


print("\n" + "="*80)
print("MOUSE GENOME DE BRUIJN GRAPH ASSEMBLY")
print("Student Assignment Driver Program")
print("="*80 + "\n")

# Run the assembly pipeline
result = assemble_mouse_genome()

# Display sample contigs
print("="*80)
print("SAMPLE ASSEMBLED CONTIGS")
print("="*80 + "\n")

# create a generator object of contigs as strings (generator saves memory over a list here)
contigs = (tour_to_sequence(node_list) for node_list in result['contigs'])
i = 0    # initialize an index so we can track how many previews we've printed
for contig in contigs:
    preview = contig[:100] + "..." if len(contig) > 100 else contig
    print(f">contig_{i+1} length={len(contig)}")
    print(preview)
    print()

    # increment our index by 1
    i += 1
    if i == 5:
        # since we've printed contigs 1-5, we can stop iterating over our generator
        break

print("="*80)
print("✓ ASSEMBLY COMPLETE")
print("="*80)
print(f"\nResults saved to:")
print(f"  • mouse_assembly.fasta - {result['stats']['num_contigs']:,} assembled contigs")
print(f"  • mouse_assembly_stats.txt - Detailed assembly statistics")
print(f"\nKey metrics:")
print(f"  • Total assembly: {result['stats']['total_length']:,} bp")
print(f"  • N50: {result['stats']['n50']:,} bp")
print(f"  • Longest contig: {result['stats']['longest_contig']:,} bp")
print(f"  • Coverage: {result['coverage']:.1f}x")
print()


