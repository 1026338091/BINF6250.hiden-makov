from collections import defaultdict
import random

def _nodes_to_path(nodes):
    # list of (k-1)-mers -> contig string
    return nodes[0] + "".join(n[-1] for n in nodes[1:])

def _eulerian_walk_recursive(graph_to_consume, startnode):
    # warning!!! recursion will fail on large contigs due to recursion limit.
    # just here for posterity
    path = []

    def search(currnode):
        while graph_to_consume[currnode]:
            nextnode = graph_to_consume[currnode].pop()
            search(nextnode)
        path.append(currnode)
    
    search(startnode)

    path.reverse()
    return path

def _eulerian_walk(graph_to_consume, startnode):
    # iterative version to avoid recursion error

    stack = [startnode]
    path = []

    while stack:
        currnode = stack[-1]
        if graph_to_consume[currnode]:
            nextnode = graph_to_consume[currnode].pop()
            stack.append(nextnode)
        else:
            path.append(stack.pop())

    path.reverse()
    return path

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
        kmers_added = 0

        for read in reads:
            read = read.strip().upper()
            reads_total += 1

            if len(read) < self.k:
                ignored_short += 1
                continue # escapes reads loop

            if self.ignore_ambiguous and any(b not in "ACGT" for b in read):
                ignored_amb += 1
                continue # escapes reads loop

            for i in range(len(read) - self.k + 1):
                # loops through kmers of read
                kmer = read[i : i + self.k]

                prefix = kmer[:-1]
                suffix = kmer[1:]

                self.graph[prefix].append(suffix)

                self.balance[prefix] += 1
                self.balance[suffix] -= 1

                kmers_added += 1

        self.startnodes = [n for (n, b) in self.balance.items() if b > 0]

        edges = sum(len(adj) for adj in self.graph.values())
        nodes = len(set(self.balance.keys()) | set(self.graph.keys()))
        bal_pos = sum(1 for b in self.balance.values() if b > 0)
        bal_neg = sum(1 for b in self.balance.values() if b < 0)
        bal_zero = sum(1 for b in self.balance.values() if b == 0)
        bal_abs_gt1 = sum(1 for b in self.balance.values() if abs(b) > 1)

        self.stats = {
            "reads_total": reads_total,
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
        contig_lengths = [len(_nodes_to_path(p)) for p in contigs]
        contig_lengths.sort(reverse=True)

        self.stats.update({
            "assembly_seed": seed,
            "contigs": len(contigs),
            "total_bp": sum(contig_lengths),
            "max_bp": contig_lengths[0] if contig_lengths else 0,
            "leftover_edges_after_exhaust": leftover_edges
        })

        if verbose:
            print(f"[assembly] seed = {seed}; contigs = {self.stats['contigs']}")
            print(f"[assembly] leftover edges = {leftover_edges}")

        return contigs

    def write_fasta(self, out_path, sort_by_len=True, wrap=60):

        if self.contig_lists is None:
            raise RuntimeError("assemble_contigs() has not been run yet.")
        contigs = []

        for i, path_nodes in enumerate(self.contig_lists, start=1):
            seq = _nodes_to_path(path_nodes)
            contigs.append((i, seq))

        if sort_by_len:
            contigs.sort(key=lambda x: len(x[1]), reverse=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for idx, seq in contigs:
                f.write(f">contig_{idx:06d} len={len(seq)} k={self.k}\n")
                for i in range(0, len(seq), wrap):
                    f.write(seq[i:i+wrap] + "\n")

# toy_reads_1 = [
#     "ATGGCGTACG",  # Read 1
#     "GGCGTACGTT",  # Read 2: overlaps with Read 1
#     "CGTACGTTAC",  # Read 3: overlaps with Read 2
#     "TACGTTACCA",  # Read 4: overlaps with Read 3
#     "CGTTACCATG",  # Read 5: overlaps with Read 4
#     "TTACCATGGG",  # Read 6: overlaps with Read 5
#     "ACCATGGGCC",  # Read 7: overlaps with Read 6
#     "CATGGGCCTA",  # Read 8: overlaps with Read 7
#     "TGGGCCTAAA"   # Read 9: overlaps with Read 8
# ]

# toy_k5_a = DeBruijnGraph(toy_reads_1, 5) 
# toy_k5_a.assemble_contigs(seed=123)
# toy_k5_a.write_fasta("test_k5_a")

# toy_k5_b = DeBruijnGraph(toy_reads_1, 5) 
# toy_k5_b.assemble_contigs(seed=321)
# toy_k5_b.write_fasta("test_k5_b")

# toy_k5_c = DeBruijnGraph(toy_reads_1, 5)  
# toy_k5_c.assemble_contigs(seed=12243435435345465345) # another seed that does the same thing as a
# toy_k5_c.write_fasta("test_k5_c")

# print(toy_k5_a.contig_lists == toy_k5_b.contig_lists) # False
# print(toy_k5_a.contig_lists == toy_k5_c.contig_lists) # True

# ambiguous_and_too_short = [
#     "ATGNCGTACG",
#     "GGCGTACRTT",
#     "CGTACGTTAC",
#     "TACGTTACCA",
#     "ATCG"
# ]

# amb = DeBruijnGraph(ambiguous_and_too_short, 5) 
# amb.assemble_contigs(seed=123)
# amb.write_fasta("test")

import gzip

def iter_fastq_seqs_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        i=0
        while True:
            i +=1
            header = fh.readline()
            if not header:
                return  # EOF
            if i > 100000000:
                return 

            seq = fh.readline()
            plus = fh.readline()
            qual = fh.readline()

            yield seq.strip()

reads_iter = iter_fastq_seqs_gz("raw.fq.gz")
dbg = DeBruijnGraph(reads_iter, k=149, ignore_ambiguous=True, verbose=True)
dbg.assemble_contigs(seed=123, verbose=True)
dbg.write_fasta("contigs.fasta")