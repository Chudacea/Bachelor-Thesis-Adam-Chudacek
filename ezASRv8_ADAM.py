"""
Script written for ezasr
by gargs on 08.12.22 at 14:43
in PyCharm Pro (macOS)
for bugs contact: sriram.garg@mpi-marburg.mpg.de
"""

import os
import shutil
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Bio import SeqIO
from ete3 import Tree
from collections import defaultdict
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import (MultipleLocator)
from tqdm import tqdm
from argparse import ArgumentParser, RawTextHelpFormatter


def hms_string(sec_elapsed):
    """
    Computes a human-readable hms string
    :param sec_elapsed:
    :return:
    """
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def args_parser():
    """
    Parse command line arguments.
    Returns parser with command line argument values as attributes.
    """
    description = """Automates the ASR pipeline followed by gap reconciliation using Maximum Parsimony. """ \
                  """Uses IQ-Tree and PastML. v6.0 created 15.09.2022"""
    epilog = "For help on each of the Sub-commands just type the subcommand followed by -h. " \
             "For e.g. ezASRv#.py complete -h " \
             "\n" \
             "Complete run looks like: ezASRv#.py [--auto] complete -a <Alignment> -t <Rooted Tree>" \
             "-m <IQ-Tree Model> [-seed ##]"
    parser = ArgumentParser(description=description, formatter_class=RawTextHelpFormatter,
                            usage=None, epilog=epilog)
    # Install parser
    parser.add_argument('--install_help',
                        help='Flag to print install instructions',
                        required=False, action='store_true')

    # Subcommands subparsers
    subparsers = parser.add_subparsers(help='Sub-commands', required=False, dest='subcommands')

    ####################################################################################################################
    # Plot sub parsers
    plot = subparsers.add_parser(
        'plot',
        help='Plot posterior probabilities')
    plot.add_argument(
        '-rs', '--reconciled_sequences',
        help='File with gap reconciled ancestral sequences in fasta format',
        required=True, type=str, metavar="\b", dest='rs')
    plot.add_argument(
        '-s', '--state_file',
        help='Ranked state file from IQ-tree results. Not the original .state file!',
        required=True, type=str, metavar="\b", dest='sf')

    ####################################################################################################################
    # Complete run sub parsers
    complete_run = subparsers.add_parser(
        'complete',
        help="Perform the complete pipeline")

    # Complete run arguments
    complete_run.add_argument('--auto',
                              help='For an automatic run without raw input requirement. Call BEFORE subcommands',
                              required=False, action='store_true')
    complete_run.add_argument(
        '-a', '--alignment',
        help='Input alignment file in fasta format.',
        required=True, type=str, metavar="\b")
    complete_run.add_argument(
        '-af', '--format',
        help='Format of the alignment file. Will only accept formats readable by BioPython 1.7 or '
             'higher. Default is FASTA. Better to stick with it',
        required=False, default='fasta', type=str, metavar="\b")
    complete_run.add_argument(
        '-t', '--tree',
        help='Pre-calculated Rooted Tree file in Newick format readable by ETE3',
        required=True, type=str, metavar="\b")
    complete_run.add_argument(
        '-m', '--model',
        help='Model used for building the tree/ Model to be used for ASR in the IQ-Tree Command. '
             'If no Model is provided will run ModelFinder Plus on the alignment before proceeding.'
             ' Will require considerable time',
        required=False, default='MFP', type=str, metavar="\b")
    complete_run.add_argument(
        '--noded',
        help='Flag to indicate if the tree has node names or if they need to be labelled. '
             'If provided will proceed with the ASR. Unlabelled nodes will get a node label'
             'from IQ-Tree. '
             'Will result in a error at the PastML step.',
        required=False, action='store_true', default=False)
    complete_run.add_argument(
        '-seed',
        help='Seed value for reproducible IQ-tree run.',
        required=False, default=856792)
    complete_run.add_argument(
        '-nt',
        help='Flag for telling it is a nt alignment',
        required=False, action='store_true')
    complete_run.add_argument(
        '-o', '--output_fldr',
        help='Output folder name. Default is a asr-results folder where the tree is. For multiple runs this gets '
             'overwritten. So make sure to change the name',
        required=False, type=str, metavar="\b")

    ####################################################################################################################
    # Reconcile only subparsers
    reconcile_only_subparser = subparsers.add_parser(
        'reconcile_only',
        help='Only perform the gap reconciliation with an edited PastML like combined_ancestral_states.tab file')

    # Reconcile only arguments
    reconcile_only_subparser.add_argument(
        '-gt', '--gaps_table',
        help='A file like the combined_ancestral_states.tab file that is generated by PastML.\
              The file should be a tab separated file with the first colum as the node name and\
              then rest as gaps. They can be filled or not filled. i.e. the ambiguous version can\
              have the same node name and just a 1/0 on the site that you need to change or \
              completely unique. Try to stick the version numbering to *_v[0-*]',
        required=True, type=str, dest='gaps', metavar="\b")
    reconcile_only_subparser.add_argument(
        '-iqt', '--iqtree_recommended_sequences',
        dest='iqt', required=True, metavar="\b")
    reconcile_only_subparser.add_argument(
        '-o', '--output_fldr',
        help='Output folder name. Default is a reconcile-only-results folder where the tree is. '
             'For multiple runs this gets overwritten. So make sure to change the name',
        required=False, type=str, metavar="\b")

    ####################################################################################################################
    # Alternatives subparsers
    alternatives_subparser = subparsers.add_parser(
        'alternatives',
        help='Goes through the state ranked file and reconciles with pastml gaps to give 2nd best ancestors\
                (AA/NT with the 2nd highest pp across all sites for each node)')

    # Alternatives arguments
    alternatives_subparser.add_argument(
        '-gt', '--gaps_table',
        help='The combined_ancestral_states.tab file that is generated by PastML.\
               The file should be a tab separated file with the first colum as the node name and\
               then rest as gaps.',
        required=True, type=str, dest='gaps', metavar="\b")
    alternatives_subparser.add_argument(
        '-s', '--state_ranked_file',
        help='Ranked state file available after a complete run',
        dest='state', required=True, metavar='\b')
    alternatives_subparser.add_argument(
        '-t', '--threshold',
        help='The threshold below which the second best aa will be chosen. Will not choose aa with pp less than 0.2\
              Default is 0.8',
        required=False, default=0.8, metavar='\b')
    alternatives_subparser.add_argument(
        '-o', '--output_fldr',
        help='Output folder name. Default is a alternatives-results folder where the tree is. '
             'For multiple runs this gets overwritten. So make sure to change the name',
        required=False, type=str, metavar="\b")
    return parser


def generate_results_folder(infolder, resfoldername='res', resfolder=None):
    """
    Creates a results folder to store results in. If no specific path is
    provided a 'res' folder is created in the same place as the inputfolder.
    :param infolder: Input folder where to make the results folder
    :param resfoldername: name of the results folder. default is res
    :param resfolder: If provided a path will simply make this folder
    :return: path of the results folder
    """

    # Getting the results folder in order

    if resfolder:
        res_folder = resfolder
        if not os.path.exists(res_folder):
            # print("Results folder does not exist")
            os.makedirs(res_folder, exist_ok=True)
            # print("Created folder %s" % res_folder)
        # print("Saving results in %s" % res_folder)
    else:
        res_folder = os.path.join(infolder, resfoldername)
        if not os.path.exists(res_folder):
            # print("Results folder not provided")
            os.makedirs(res_folder)
        # print("Created folder %s" % res_folder)
        # print("Saving results in %s" % res_folder)
    return res_folder


def check_if_rooted(treefile):
    """
    Checks if there are any nodes with more than 2 children. In all likelihood if a rooted
    tree has been provided these are polytomies. If not the get_tree_root should get the root node.
    :param treefile:
    :returns tree if no polytomies were detected and None if there were.:
    """
    tree = Tree(treefile, format=1, quoted_node_names=True)
    # Set root name (ideally if the tree is rooted with newick format
    tree.get_tree_root().name = 'root'

    # Check for polytomies (i.e. nodes with more than two children and will NOT resolve them by randomly assigning
    # them branch length zero)
    for node in tree.traverse('preorder'):
        if not node.is_leaf() and node.name != 'root':
            if len(node.get_children()) > 2:
                print("Polytomies detected. Exiting")
                return None

    # If no polytomies are detected will return the original tree with the root labelled as root
    return tree


def label_nodes(treefile, resfolder, nameprefix='Anc_'):
    """
    Checks for polytomies, so it is important that it passes check_if_rooted function. Does an isinstance-check of Tree
    class. Names unlabelled Nodes. If noded tree is given nothing is relabelled. Please check if errors occur.
    :param treefile: Newick tree file relabelled by ete3. uses the format=0 option.
    :param resfolder: Results folder to store the tree file with internal nodes labelled.
    :param nameprefix: Preferred prefix for ancestral nodes.
    :return: The labelled tree filename (full path)
    """

    tree = check_if_rooted(treefile)
    # Check if tree is a Tree
    if isinstance(tree, Tree):

        # Counter for nodes
        node_counter = 1
        for node in tree.traverse('preorder'):
            if not node.is_leaf():
                # Don't change name if name is root or has root.
                if 'root' in node.name.lower():
                    pass
                # Don't change name if name is already present and is longer than zero
                elif len(node.name) > 0:
                    pass
                    # node.name = node.name
                # Change the name according to counter. The unlabelled nodes will be numbered starting from 1
                elif node.name != ' ':
                    node.name = '{}{}'.format(nameprefix, node_counter)
                    node_counter = node_counter + 1

    # Output tree name
    out_tree_fn = f"{os.path.splitext(os.path.basename(treefile))[0]}.nodetree"
    out_tree = os.path.join(resfolder, out_tree_fn)

    # Write tree with output tree name
    tree.write(format=1, outfile=out_tree, )

    return os.path.abspath(out_tree)


def run_iqtree_asr(resfolder, alignmentfile, treefile, model='MFP', seed=856792, command=None, auto=False):
    """
    A wrapper for running the IQtree command and folders. If the command has to be changed type the command directly
    will not autocomplete within python. Also, very risky since it runs with shell=True.
    :param resfolder:
    :param alignmentfile:
    :param treefile:
    :param model:
    :param seed:
    :param command:
    :param auto:
    :return:
    """

    start_time = time.time()

    # Try to create a tree results folder. Exception needed if folder already exists.
    results_dir = generate_results_folder(infolder=os.path.dirname(alignmentfile),
                                          resfolder=resfolder, resfoldername='iqtree_res')

    # Results prefix for IQ-tree
    results_prefix = os.path.join(results_dir, os.path.splitext(os.path.basename(alignmentfile))[0])

    if command is None:
        command = ' '.join(
            ['./iqtree3', '-s', alignmentfile, '-m', model, '-keep_empty_seq',
             '-te', treefile,
             '-asr', '-nt AUTO','-ntmax 18','-seed', str(seed), '-pre', results_prefix])
    print(f"Running IQ-tree with command:\n\t{command}")

    # Checkpoint to proceed if not AUTO. One can change command when needed. Will not autofill though
    if not auto:
        response = input('Proceed? [(N)o/(Y)es/(C)hange]')
    else:
        response = 'y'

    if response.lower()[0] == 'y':
        subprocess.run(command, shell=True)
        state_file = '{}.{}'.format(results_prefix, 'state')
        print(f"Finished Running IQ-tree in {hms_string(time.time() - start_time)} seconds")
        return state_file

    elif response.lower()[0] == 'c':
        new_command = input("Please type updated IQ-tree command: ")
        if command.split()[0] == 'iqtree':
            subprocess.run(new_command, shell=True)
            state_file = '{}.{}'.format(results_prefix, 'state')
            print(f"Finished Running IQ-tree in {hms_string(time.time() - start_time)} seconds")
            return state_file
        else:
            print("Illegal command given. try iqtree [-options]")
    elif response.lower()[0] == 'n':
        print(f"Exiting. Bye! Wasted {hms_string(time.time() - start_time)} seconds")
        quit()
    else:
        quit()

    return None


def rank_state_file(resfolder, statefileiqtree, nt=False):
    # Re-orders the original IQ-tree statefile in increasing order. The aminoacids listed in the following list are
    # ABSOLUTELY crucial. changing the order will not give an error but will be read in the same order. DO NOT CHANGE!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    iqtree_aa_list = ['p_A', 'p_R', 'p_N', 'p_D', 'p_C', 'p_Q', 'p_E', 'p_G', 'p_H', 'p_I', 'p_L', 'p_K', 'p_M', 'p_F',
                      'p_P', 'p_S',
                      'p_T', 'p_W', 'p_Y', 'p_V']
    iqtree_nt_list = ['p_A', 'p_C', 'p_G', 'p_T']
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    rank_dict = defaultdict(dict)

    with open(statefileiqtree, 'r') as f:
        for line in f:
            if '#' not in line and 'Node' not in line:
                cols = (line[:-1].split('\t'))
                node = cols[0]
                site = cols[1]
                rank_dict[node][site] = []  # {}
                if nt:
                    for aa, p_NT in zip(iqtree_nt_list, cols[3:]):
                        rank_dict[node][site].append('{}_#_{}'.format(aa, p_NT))
                else:
                    for aa, p_AA in zip(iqtree_aa_list, cols[3:]):
                        rank_dict[node][site].append('{}_#_{}'.format(aa, p_AA))

    sorted_ranked_dict = defaultdict(dict)
    for node in rank_dict:
        for site in rank_dict[node]:
            sorted_ranked_dict[node][site] = []

            # Sort the list based on pAA/pNT (hence the split)
            s = rank_dict[node][site]
            s.sort(key=lambda x: float(x.split('_#_')[-1]), reverse=True)
            sorted_ranked_dict[node][site] = s

    # Write the results to file
    res_file = (f"{statefileiqtree}_ranked")

    res = open(res_file, 'w+')
    for node in sorted_ranked_dict:
        for site in (sorted_ranked_dict[node]):
            wline = "{}\t{}\t{}\n".format(node, site, '\t'.join(sorted_ranked_dict[node][site]))
            res.write(wline)
    res.close()

    return res_file


# Requires rank_state_file
def rankfile_to_sequences(resfolder, statefileiqtree, nt):
    """
    Gets the recommended sequences from the iqtree state file. Nothing more.
    To get the ranked Amino acids write another function. Calls ranked functions

    :param nt:
    :param resfolder:
    :param statefileiqtree:
    :return:
    """
    # Produces the ranked file where the amino acids/nucleotides
    # are ranked by probability (across columns per node per site)
    ranked_state_file = rank_state_file(resfolder, statefileiqtree, nt=nt)

    node_anc_seq_dict = defaultdict(list)
    with open(statefileiqtree, 'r') as f:
        for line in f:
            if '#' not in line and 'Node' not in line:
                cols = (line.split('\t'))
                k = cols[0]
                v = cols[2]
                node_anc_seq_dict[k].append(v)

    res_name = os.path.join(resfolder, f"{os.path.splitext(os.path.basename(statefileiqtree))[0]}_ancestors.fasta")
    res = open(res_name, 'w+')
    for node in node_anc_seq_dict:
        seq = ''.join(node_anc_seq_dict[node])
        line = '>{}\n{}*\n'.format(node, seq)
        res.write(line)
    res.close()

    return res_name, ranked_state_file


def make_binary_alignment(resfolder, alnfile, alignment_format='fasta'):
    """
    Takes an alignment file (default format is fasta but can accept all formats Biopython.SeqIO can read) and
    converts all the positions to 1s and gaps to 0s. Will then store these values as a binary alignment file and
    also as a dataframe as output.

    :param resfolder: Path to the folder where the results will be stored.
    :param alnfile: Path to alignment file
    :param alignment_format: Format in which the alignment file is stored.
    :return: path of the csv file.
    """

    # Get the basename of alignment file
    aln_fileprefix = os.path.basename(alnfile).split('.')[0]

    # Generate the name of the results file. In this case with .binary_aln extension
    alnres_path = os.path.join(resfolder, '{}.binary_aln'.format(aln_fileprefix))
    dfres_path = os.path.join(resfolder, '{}_binary.csv'.format(aln_fileprefix))

    # Open res file
    res = open(alnres_path, 'w+')
    binary_dict = defaultdict(list)

    # Read the sequences
    sequences = SeqIO.index(alnfile, format=alignment_format)
    if len(sequences) > 0:
        for ids in sequences:
            for position in sequences[ids].seq:
                if position == '-':
                    binary_dict[ids].append('0')
                else:
                    binary_dict[ids].append('1')
            line = '>' + ids + '\n' + ''.join(binary_dict[ids]) + '\n'
            res.write(line)

    # Close result file
    res.close()

    # Dirty way to get length of alignment (amino acids)
    # Important for alphanumeric sorting
    col_names = {}
    for aa in range(len(binary_dict[ids])):
        col_names[aa] = 'Site_{}'.format(f'{aa:05}')

    # Saving as a dataframe
    df = pd.DataFrame.from_dict(binary_dict, orient='index')
    df.rename(columns=col_names, inplace=True)
    df.to_csv(dfres_path, sep='\t')

    return dfres_path


def fill_combined_state_file(combinedstatefile):
    # This will read the ambiguities in pastML recreations and create a new combined gap state file. Since the default
    # output of pastML leaves blanks.

    # Create new file to store the filled state file
    fill_combined_state_filename = '{}_filled.tab'.format(combinedstatefile.split('.')[0])

    # Load original pastML output to a Dataframe and then to a dict. Shortcut to create a nested dictionary.
    df = pd.read_csv(combinedstatefile, sep='\t', header=0, index_col=0)
    df.drop(labels=['root'], inplace=True)
    asr_dict = (df.groupby(by=df.index).apply(lambda x: x.to_dict(orient='records'))).to_dict()

    # New dictionary
    new_asr_dict = {}
    for ancestor in asr_dict:
        if len(asr_dict[ancestor]) > 1:
            # Create a temporary dictionary for ambiguous ancestors
            ambi_nodes_dict = {}
            # Rename ancestors with ambiguous values with count. Stars with v0
            for count, g in enumerate(range(len(asr_dict[ancestor]))):
                key = f"{ancestor}_v{count}"
                value = asr_dict[ancestor][count]
                ambi_nodes_dict[key] = value

            for ambi_nodes in ambi_nodes_dict:
                for site in ambi_nodes_dict[ambi_nodes]:
                    # Nan is default in the dataframe for no values.This replaces Nans with the value of the 0th version
                    if 'nan' in str(ambi_nodes_dict[ambi_nodes][site]):
                        v0_ambinode = f"{ambi_nodes[:-3]}_v0"
                        ambi_nodes_dict[ambi_nodes][site] = float(ambi_nodes_dict[v0_ambinode][site])
            new_asr_dict.update(ambi_nodes_dict)
        else:
            new_asr_dict[ancestor] = (asr_dict[ancestor][0])
    # Convert new dictionary back to dataframe for easy writing into a new file.
    df2 = pd.DataFrame.from_dict(new_asr_dict, orient='index')
    df2.to_csv(fill_combined_state_filename, sep='\t')

    return new_asr_dict


def run_pastml(rootedtreefile, binarycsvfile):
    # Running PastML
    command = ' '.join(['pastml', '-t', rootedtreefile, '-d', binarycsvfile, "-s '\t' --prediction_method ACCTRAN"])

    print(f"Running PastML with command:\n\t{('%r' % command)[1:-1]}")
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    p.stdout.close()
    p.wait()

    return rootedtreefile


def parse_pastml(resfolder, rootedtreefile, combinedstatefile=None):
    # Parsing Output
    # Guessing resultsfoldername

    if not combinedstatefile:
        pastml_results_folder = '{}_pastml'.format(os.path.join(
            str(os.path.dirname(rootedtreefile)),
            '.'.join(os.path.basename(rootedtreefile).split('.')[:-1])))
        combined_state_file = os.path.join(pastml_results_folder, 'combined_ancestral_states.tab')
    else:
        combined_state_file = combinedstatefile

    df = pd.read_csv(combined_state_file, sep='\t', header=0, index_col=0)

    # PastML gives two root nodes. These are not interesting since we do not look at root reconstructions from
    # IQ-tree ASR and moreover when we do we would add a fake node as the root node and therefore the root node
    # from PastML will always be not interesting
    df.drop(labels=['root'], inplace=True)

    # Load the new dataframe into a dictionary of dictionary
    try:
        asr_dict = df.to_dict(orient='index')
    except ValueError:
        print("Detected Ambiguities. Trying GroupBy...")
        print("Will parse file and create new one with filled values")
        asr_dict = fill_combined_state_file(combined_state_file)

    # Result filename
    res_filename = os.path.join(resfolder,
                                f"{os.path.basename(rootedtreefile).split('.')[0]}_ancestral_gaps.txt")
    res = open(res_filename, 'w+')

    # Writing into the results file
    for node in asr_dict:

        # To skip first line
        if 'node' not in node:
            gap_seq = []
            for site in sorted(asr_dict[node]):
                gap_seq.append(str(int(asr_dict[node][site])))
            writeline = ('>{}\n{}\n'.format(node, (''.join(gap_seq))))
            res.write(writeline)

    # Close result file
    res.close()

    return res_filename


def reconcile_iqtree_asr(ancestralgapsfile, iqtreeancestralsequences, resfolder, resname=None, gap_format='fasta',
                         asr_format='fasta'):
    """
    Takes the ancestral gaps file generated by run_pastml of all the nodes and uses this to assign gaps to the
    ancestral sequences. Needs two multifasta type files in fasta format.

    :param resfolder: Folder where the results will be stored
    :param resname: Name of the result file.
    :param gap_format: Format of the gap multifasta file. Can take all Biopython SeqIO formats. Default is Fasta
    :param asr_format: Format of the ancestral sequences
                       multifasta file. Can take all Biopython SeqIO formats. Default is Fasta
    :param ancestralgapsfile:
    :param iqtreeancestralsequences:
    :return:
    """

    # Loading the two sequences
    iqt_anc_seq = SeqIO.index(iqtreeancestralsequences, format=asr_format)
    anc_gaps = SeqIO.index(ancestralgapsfile, format=gap_format)
    # print(anc_gaps)
    # Opening the result file
    if resname is None:
        res_name = os.path.join(resfolder,
                                f"{os.path.splitext(os.path.basename(ancestralgapsfile))[0]}_reconciled.fasta")
    else:
        res_name = os.path.join(resfolder, resname)

    res = open(res_name, 'w+')

    for node in anc_gaps:
        gap_index = []
        site_index = []
        for index, site in enumerate(anc_gaps[node].seq):
            # print(index,site)
            if site == '0':
                gap_index.append(index)
            elif site == '1':
                site_index.append(index)

        # The ambiguous nodes if present have _v1. If error occurs Could be because of names.
        # Needs a second try for alternative sequences.
        try:
            iqt_node_name = node
            iqt_anc_seq_list = list(iqt_anc_seq[iqt_node_name].seq)
            iqt_node_name_suffix = ""
        except KeyError:
            try:
                iqt_node_name = node.split('_v')[0]
                iqt_node_name_suffix = f"_{node.split('_')[-1]}"
                iqt_anc_seq_list = list(iqt_anc_seq[iqt_node_name].seq)
            except KeyError:
                try:
                    iqt_node_name = f"{node}_alt"
                    iqt_node_name_suffix = ""
                    iqt_anc_seq_list = list(iqt_anc_seq[iqt_node_name].seq)

                except KeyError:
                    # print(f"{node} not in ancestral sequences. Skipping...")
                    continue
        # Reconciliation happens now
        for i in gap_index:
            iqt_anc_seq_list[i] = '-'
        for i in site_index:
            if iqt_anc_seq_list[i] == '-':
                print(
                    f"Gap found where parsimony says AA/NT. Written in sequence as '#' "
                    f"Please verify!!! Not my problem.")
                iqt_anc_seq_list[i] = '#'
        writeline = '>{}\n{}\n'.format(f"{iqt_node_name}{iqt_node_name_suffix}", str(''.join(iqt_anc_seq_list)))
        res.write(writeline)
    res.close()

    return res_name


def plot_posterior_and_average(reconciledasrseqs, asrstatefile, resfolder=None):
    # This function is not perfect. When necessary please plot your own ancestors with the ranked file as guidance.

    # Read ancestral sequences with SeqIO
    asr_sequences = SeqIO.index(reconciledasrseqs, format='fasta')

    # Create dictionary with aminoacids and gaps.
    anc2site = defaultdict(dict)
    for ancestor in asr_sequences:
        for index, site in enumerate(str(asr_sequences[ancestor].seq), start=1):
            if '*' not in site:
                anc2site[ancestor][str(index)] = site
    #print(anc2site.keys())
    # Create state rank dict
    state_dict = defaultdict(dict)
    with open(asrstatefile, 'r') as f:
        for line in f:
            cols = line[:-1].split('\t')
            state_dict[cols[0]][cols[1]] = defaultdict(dict)
            for aaprob in cols[2:]:
                aa = aaprob.split('_')[1]
                prob = aaprob.split('_')[-1]
                state_dict[cols[0]][cols[1]][aa] = prob

    #print(state_dict.keys())
    # print(state_dict['Anc_1']['2'].keys())

    # # Create probability dictionary
    # anc2prob_wo_gaps = defaultdict(dict)
    # with open(asrstatefile, 'r') as f:
    #     for line in f:
    #         cols = line[:-1].split('\t')
    #         anc2prob_wo_gaps[cols[0]][(cols[1])] = cols[2].split('_#_')[-1]
    # #print(anc2prob_wo_gaps)
    #
    # anc2aaprob = defaultdict(dict)
    # with open(asrstatefile, 'r') as f:
    #     for line in f:
    #         cols = line[:-1].split('\t')
    #         ancestor = cols[0]
    #         site = cols[1]
    #         anc2aaprob[ancestor][site] = defaultdict(dict)
    #         for aaprob in cols[2:]:
    #             aa = aaprob.split('_')[1]
    #             prob = aaprob.split('_')[-1]
    #             anc2aaprob[ancestor][site][aa] = prob
    # #print(anc2aaprob['Anc_1'].keys())

    anc2prob = defaultdict(list)
    for anc in list(anc2site.keys()):
        ancestor = '_'.join(anc.split('_')[:2])
        for site in anc2site[anc]:
            aa = anc2site[anc][site]
            if aa != '-' and aa != '#':
                prob = (state_dict[ancestor][site][aa])
                anc2prob[anc].append(prob)

    if not resfolder:
        resfolder = os.path.dirname(reconciledasrseqs)
    res_folder = os.path.join(resfolder, 'pp_plots')

    try:
        os.makedirs(res_folder, exist_ok=True)
    except FileExistsError:
        pass

    for ancestor in tqdm(anc2prob):
        #print(ancestor)
        #print(anc2prob[ancestor])
        #print(type(anc2prob[ancestor]))

        probabilities = np.asarray(anc2prob[ancestor], dtype='float')
        counts, probs = (np.histogram(probabilities, bins=20))
        second_largest = counts[-2]
        sns.set_style('white')

        # Generate plot with multiple y axes (Trick is to generate two subplots and hide the gap)
        # Gridspec_kw dictates space between.
        f, (ax_uh, ax_bh) = plt.subplots(2, 1, sharex='all', gridspec_kw={'hspace': 0.045})

        # Plot same data on both axes
        plotb = sns.histplot(data=probabilities, bins=20, binrange=(0, 1), color='green', ax=ax_bh, stat='count')
        plotu = sns.histplot(data=probabilities, bins=20, binrange=(0, 1), color='red', ax=ax_uh, stat='count')
        plotu.set(ylabel=None)
        plotb.set(ylabel=None)

        # Zoom-in / limit the view to different portions of the data
        # ax_bh.set_xlim(0, 1.15)
        ax_bh.set_ylim(0, (round(second_largest / 10) * 10) + 10)
        # print(3*(len(probabilities)/5), len(probabilities))
        # ax_uh.set_xlim(0, 1.15)
        ax_uh.set_ylim(round(3 * (len(probabilities) / 5)), len(probabilities))

        # Remove boxes from the top and bottom graph
        ax_uh.spines['bottom'].set_visible(False)
        ax_bh.spines['top'].set_visible(False)

        # Major ticks in multiples of 50 and 5 (has to be hard coded)
        ax_uh.yaxis.set_major_locator(MultipleLocator(50))
        ax_bh.yaxis.set_major_locator(MultipleLocator(5))

        # Axes ticks display options
        ax_uh.xaxis.set_ticks_position('none')
        ax_uh.tick_params(labelbottom=False)  # don't put tick labels at the top
        ax_uh.tick_params(labeltop=False)
        ax_bh.xaxis.tick_bottom()
        ax_uh.yaxis.tick_left()
        ax_bh.yaxis.tick_left()

        # Diagonal lines
        d = .012  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax_uh.transAxes, color='k', clip_on=False)
        ax_uh.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
        ax_uh.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax_bh.transAxes)  # switch to the bottom axes
        ax_bh.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax_bh.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        # Add probability as text
        text = "Average Probability: {:.3f}".format(float(probabilities.mean()))
        avgp = AnchoredText(text, loc="upper left", frameon=False)
        ax_uh.add_artist(avgp)

        # To get common axes.
        f.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.title(f"Posterior probabilities for {ancestor}")
        plt.ylabel("Occurrence")
        plt.xlabel("Posterior probabilities")

        res_file = os.path.join(res_folder, f"pp_plot_{ancestor}.svg")

        f.figure.savefig(res_file, dpi=72)
        plt.clf()
        plt.close(f)

    return res_folder


def get_alternatives(staterankfile, resfolder, threshold=None):
    # Default variables
    if not threshold:
        threshold = 0.8

    sorted_rank_dict = defaultdict(dict)
    with open(staterankfile, 'r') as f:
        for line in f:
            cols = line[:-1].split('\t')
            node = cols[0]
            site = cols[1]
            sorted_rank_dict[node][site] = cols[2:]

    res_file = os.path.join(resfolder, f"{os.path.basename(staterankfile).split('.')[-1]}_alternatives_iqt.fasta")
    res = open(res_file, 'w+')

    for node, sites in sorted_rank_dict.items():
        sequence = []
        for site, vals in sites.items():
            first_aa_pp = vals[0].split('_#_')[-1]
            if float(first_aa_pp) < threshold:
                second_aa_pp = vals[1].split('_#_')[-1]
                if float(second_aa_pp) > 0.2:
                    # print(f"Found alternative for site: {site}, with pp {second_aa_pp} instead of {first_aa_pp} ")
                    sequence.append(vals[1][2])
                else:
                    sequence.append(vals[0][2])
            else:
                sequence.append(vals[0][2])
        wline = f">{node}_alt\n{''.join(sequence)}\n"
        res.write(wline)

    res.close()

    return res_file


def main():
    """
    Main function with all the calls needed for the ezASR to run
    :return:
    """
    # Begin
    print('Sriramajayam')
    # Initialize the start time
    start_time = time.time()

    # Initialize all the arguments
    # global args
    args = args_parser().parse_args()

    # If the function was called without any arguments then simply print the usage
    if not len(sys.argv) > 1:
        print(args_parser().print_usage())

    # If the --install_help flag is invoked then print installation help and exit the code.
    if args.install_help:
        print(install_help)
        sys.exit()

    # Commands for alternatives run
    if args.subcommands == 'alternatives':
        # Get the global results folder
        if args.output_fldr:
            results_folder = generate_results_folder(infolder=os.path.dirname(args.state),
                                                     resfoldername=args.output_fldr, resfolder=args.output_fldr)
        else:
            results_folder = generate_results_folder(infolder=os.path.dirname(args.state),
                                                     resfoldername="ezASR_alternatives_results")

        alt_sequences = get_alternatives(resfolder=results_folder, staterankfile=args.state, threshold=args.threshold)
        res_name = f"{os.path.basename(alt_sequences).split('.')[0]}_reconciled.fasta"
        reconciled_seq_file = reconcile_iqtree_asr(ancestralgapsfile=args.gaps, resfolder=results_folder,
                                                   iqtreeancestralsequences=alt_sequences, resname=res_name)
        print(f"Finished reconciling alternative ancestors...")
        print(f"Plotting posterior probabilities for alternatives...")
        plot_folder = plot_posterior_and_average(reconciledasrseqs=reconciled_seq_file,
                                                 asrstatefile=args.state)
        print(f"Completed Analysis.\n"
              f"\tFinal ancestral sequences: {reconciled_seq_file}\n "
              f"\tPlots of posterior probabilites for each ancestor: {os.path.abspath(plot_folder)}\n")

    # Commands for reconcile only run
    if args.subcommands == 'reconcile_only':
        # Get the global results folder
        if args.output_fldr:
            results_folder = generate_results_folder(infolder=os.path.dirname(args.iqt),
                                                     resfoldername=args.output_fldr, resfolder=args.output_fldr)
        else:
            results_folder = generate_results_folder(infolder=os.path.dirname(args.iqt),
                                                     resfoldername="ezASR_reconcile_only_results")

        ancestral_gaps_file = parse_pastml(rootedtreefile="reconcile_only", combinedstatefile=args.gaps,
                                           resfolder=results_folder)
        print(f"Finished parsing modified PastML table. Reconciling IQ-tree ASR suggestions with gaps...")
        reconciled_seq_file = reconcile_iqtree_asr(resfolder=results_folder, ancestralgapsfile=ancestral_gaps_file,
                                                   iqtreeancestralsequences=args.iqt)

        print(f"Completed Analysis.\n"
              f"\tFinal ancestral sequences saved in: {reconciled_seq_file}\n ")
        print("Will not plot posterior probabilities. Call program again with 'plot'. Exiting now. Bye!")

    # Commands for plot only
    if args.subcommands == 'plot':
        print(f"Plotting posterior probabilities...")
        plot_folder = plot_posterior_and_average(reconciledasrseqs=args.rs,
                                                 asrstatefile=args.sf)
        print(f"Completed Plotting.\n"
              f"\tPlots of posterior probabilites for each ancestor: {os.path.abspath(plot_folder)}\n")

    # Commands for the complete run.
    if args.subcommands == 'complete':
        # Get the global results folder
        if args.output_fldr:
            results_folder = generate_results_folder(infolder=os.path.dirname(args.tree),
                                                     resfoldername=args.output_fldr,
                                                     resfolder=args.output_fldr)
            print(results_folder)
        else:
            results_folder = generate_results_folder(infolder=os.path.dirname(args.tree), resfoldername="ezASR_results")
            print(results_folder)

        # If node flag is provided then declare noded_tree file variable
        if not args.noded:
            print("You say the tree has no Node labels so labelling the nodes. Still Expecting tree to be rooted. "
                  "Will give errors if there are polytomies. "
                  "Illegal to resurrect root!!")
            noded_treefile = label_nodes(treefile=args.tree, resfolder=results_folder)
        else:
            print("Assuming required nodes and root are labelled and proceeding with ASR")
            # Move the provided tree into the results folder
            noded_treefile = shutil.move(args.tree, os.path.join(results_folder, args.tree))

        # Run IQ-Tree with the ASR option.
        state_file = run_iqtree_asr(resfolder=results_folder, alignmentfile=args.alignment, treefile=noded_treefile,
                                    model=args.model, seed=args.seed, auto=args.auto)

        # Get the ancestral sequences from IQ-Tree state file. Also rank the state file
        ancestral_sequnces, ranked_state_file = rankfile_to_sequences(resfolder=results_folder,
                                                                      statefileiqtree=state_file, nt=args.nt)

        # Convert the alignemnt file into a binary alignment file for processing gaps in pastML
        binary_csv = make_binary_alignment(resfolder=results_folder,
                                           alnfile=args.alignment, alignment_format=args.format)
        print("Finished Binary alignment. Proceeding with PastML...")

        # Run and parse the pastML output
        if args.noded:
            ancestral_gaps_file = parse_pastml(resfolder=results_folder,
                                               rootedtreefile=run_pastml(rootedtreefile=args.tree,
                                                                         binarycsvfile=binary_csv))
        else:
            ancestral_gaps_file = parse_pastml(resfolder=results_folder,
                                               rootedtreefile=run_pastml(rootedtreefile=noded_treefile,
                                                                         binarycsvfile=binary_csv))
        print(f"Finished PastML. Reconciling IQ-tree ASR suggestions with gaps...")

        # Reconciling gaps with ancestral sequences.
        reconciled_seq_file = reconcile_iqtree_asr(ancestralgapsfile=ancestral_gaps_file,
                                                   iqtreeancestralsequences=ancestral_sequnces, resfolder=results_folder)
        print(f"Finished filling gaps. Plotting posterior probabilities...")

        # Plot the posterior probabilities
        plot_folder = plot_posterior_and_average(reconciledasrseqs=reconciled_seq_file,
                                                 asrstatefile=ranked_state_file, resfolder=results_folder)

        print(f"Completed Analysis.\n"
              f"\tFinal ancestral sequences: {reconciled_seq_file}\n "
              f"\tTree with labelled nodes: {noded_treefile}\n "
              f"\tProbabilities of the other states:{ranked_state_file}\n"
              f"\tPlots of posterior probabilites for each ancestor: {os.path.abspath(plot_folder)}\n"
              f"Please verify node labels and root, polytomy errors (indicating tree was un-rooted) and "
              f"gap reconciliation errors if thrown.\nThe sequences used the best ranked AA. Other AAs are stored in "
              "the rank file.")

    print("Exiting now. Bye!")
    print(f"Finished in {hms_string(time.time() - start_time)} seconds")

    return


if __name__ == '__main__':
    main()

# End

install_help = """
    ****INSTALLATION HELP****

    Requires Several programs and python packages to run. 
    Can be installed with Conda or individually but made available to PATH. 
    conda installations: iqtree, mafft, biopython, pip, tqdm, matplotlib, seaborn
    pip installations: pastml (will install ete3)

    Commands to install:
    # Install Anaconda (read conda documentation at https://docs.anaconda.com/anaconda/install/index.html)
    # Create New conda Environment. <env_name> is whatever name you choose.
        conda create -n <env_name> -y
    # Activate conda environment (Might have to run conda init and re-start shell before) 
        conda activate <env_name>                
    # Install packages
        conda install -n <env_name> -c bioconda pip iqtree mafft biopython tqdm seaborn matplotlib numpy 
        pip install pastml 

    # Above commands are required only once. Once the environment has been installed only do the following
    # Run ezASR in the new activated environment.
        conda activate <env_name>
        ezasr_iqtree.py -a <Alignment> -t <Rooted Tree> -m <IQ-Tree Model>
        conda deactivate <env_name>
    *************************
    """
