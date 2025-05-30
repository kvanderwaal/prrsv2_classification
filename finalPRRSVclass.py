import pandas as pd
import numpy as np
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import skops.io as sio
from io import BytesIO
import requests
import argparse

# Reference sequence (603 nt, no gaps)
REFERENCE = (
    "atgttggggaaatgcttgaccgcgggctgttgctcgcaattgctttttttgtggtgtatcgtgccgttctgttttgttgcgctcgtcaacgccaacaacaacagcagctcccatttacagttgatttataacctgacgatatgtgagctgaatggcacagattggctaaataaaaaatttgattgggcagtggagacttttgtcatctttcctgtgttgactcacattgtctcctatggcgccctcaccaccagccatttccttgacacagtcggcctgatcactgtgtctaccgccggatattatcacgggcggtatgtcttgagtagcatttacgctgtctgtgccctggctgcgttgacttgcttcgtcattaggttagcaaaaaattgcatgtcctggcgctactcatgtaccagatataccaactttcttctggacaccaagggcaaactctatcgttggcggtcgcccgtcatcatagagaaagggggtaaagttgaggtcgaaggtcacctgatcgacctcaaaagagttgtgcttgatggttccgcggcaacccctgtaaccaaagtttcagcggaacaatggggtcgtccttag"
)

def align_to_reference(seq, reference):
    """
    Align a single sequence to the reference using global alignment.
    The reference will not be modified (no gaps inserted), only the query sequence will get gaps or be trimmed.
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score = 2
    aligner.mismatch_score = -2

    # Only allow gaps in the query, not in the reference
    aligner.target_open_gap_score = -1e8
    aligner.target_extend_gap_score = -1e8

    # Run alignment
    alignments = aligner.align(reference, seq)
    best = alignments[0]
    aligned_ref = best.aligned[0]
    aligned_query = best.aligned[1]

    # Build the gapped sequence for the query such that
    # it matches the reference (603 nt, with gaps inserted as needed)
    ref_idx = 0
    query_idx = 0
    result = []

    for (ref_start, ref_end), (query_start, query_end) in zip(aligned_ref, aligned_query):
        # Fill any gaps in the reference (should not be needed)
        while ref_idx < ref_start:
            result.append('-')
            ref_idx += 1
        # Insert aligned bases
        while ref_idx < ref_end and query_idx < query_end:
            result.append(seq[query_idx])
            ref_idx += 1
            query_idx += 1
    # Fill any trailing gaps
    while len(result) < len(reference):
        result.append('-')
    # If the sequence is longer than the reference, we cut it
    return ''.join(result)[:len(reference)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimating PRRSV-2 phylogenetic variant using random forest')
    parser.add_argument('-s', '--seqali', type=str, required=True, help="PRRSV-2 multiple sequence alignment with length of 603 nt")
    parser.add_argument('-o', '--out', type=argparse.FileType('w'), required=True, help="Name or full path of classification report in .csv format")
    args = parser.parse_args()

    url0 = 'https://github.com/kvanderwaal/prrsv2_classification/raw/main/randomCV10RF.skops?download='
    url1 = BytesIO(requests.get(url0).content)
    model1 = sio.load(url1, trusted=True)

    url2 = 'https://github.com/kvanderwaal/prrsv2_classification/raw/main/sublinRF.skops?download='
    url3 = BytesIO(requests.get(url2).content)
    model2 = sio.load(url3, trusted=True)

    # Parse all sequences from the input FASTA
    with open(args.seqali) as fp:
        records = [
            {
                'name': str(record.description),
                'sequence': str(record.seq).lower().replace('\n', '').replace('\r', '')
            }
            for record in SeqIO.parse(fp, "fasta")
        ]

    # Separate sequences: those with ungapped length = 603 and those with != 603
    seq_gapped = []
    seq_to_align = []
    for rec in records:
        ungapped_len = len(rec['sequence'].replace("-", ""))
        if ungapped_len == 603:
            seq_gapped.append(rec)
        else:
            seq_to_align.append(rec)

    # Align sequences that need it
    aligned_records = []
    for rec in seq_to_align:
        aligned_seq = align_to_reference(rec['sequence'].replace("-", ""), REFERENCE)
        aligned_records.append({'name': rec['name'], 'sequence': aligned_seq})

    # For sequences already gapped but of correct length, check for length
    for rec in seq_gapped:
        # If length is not 603, pad or trim as needed
        seq = rec['sequence']
        if len(seq) < 603:
            seq = seq.ljust(603, '-')
        elif len(seq) > 603:
            seq = seq[:603]
        aligned_records.append({'name': rec['name'], 'sequence': seq})

    # Now all entries in aligned_records have length 603 (with possible gaps)
    d = pd.DataFrame.from_records(aligned_records)
    d['sequence'] = d['sequence'].apply(lambda x: x.lower())
    d = d.set_index('name')
    d = d['sequence'].apply(lambda x: pd.Series(list(x)))
    d = d.rename(columns={x: y for x, y in zip(d.columns, range(1, len(d.columns) + 1))})
    d = d.add_prefix('p.')
    d[~d.isin(['a', 't', 'c', 'g', '-'])] = '-'
    d = d.replace('-', np.nan)

    for column in d.columns:
        d[column] = d[column].fillna(d[column].mode()[0])

    a = ['a']
    t = ['t']
    c = ['c']
    g = ['g']

    d.loc[len(d.index)] = np.repeat(a, len(d.columns))
    d.loc[len(d.index)] = np.repeat(t, len(d.columns))
    d.loc[len(d.index)] = np.repeat(c, len(d.columns))
    d.loc[len(d.index)] = np.repeat(g, len(d.columns))
    col = list(d.columns)
    dum_d = pd.get_dummies(d, columns=col)
    dum_d = dum_d.iloc[:-4]

    base_feat = dum_d.loc[:, dum_d.columns.isin(model1.feature_names_in_)]
    base_pred = model1.predict_proba(base_feat)
    col = (model1.classes_).tolist()
    ind = list(base_feat.index)
    base_pred = pd.DataFrame(base_pred, columns=col, index=ind)
    base_pred['var'] = base_pred.apply(lambda x: list(zip(x.index[x > 0], x[x > 0])), axis=1)
    base_pred['id'] = base_pred.index
    lstsort = [(row.id, sorted(row.var, key=lambda x: (-x[1], x[0]))) for row in base_pred.itertuples()]
    base_prob = pd.DataFrame(lstsort, columns=['id', 'base_var_prob'])

    new_df = base_prob.explode('base_var_prob').reset_index(drop=True)
    new_df[['base_var', 'base_prob']] = pd.DataFrame(new_df['base_var_prob'].tolist(), index=new_df.index)
    new_df = new_df[['id', 'base_var', 'base_prob']]
    new_df['idx'] = new_df.groupby('id').cumcount() + 1
    new_df = new_df.pivot_table(index=['id'], columns='idx', values=['base_var', 'base_prob'], aggfunc='first')
    new_df = new_df.sort_index(axis=1, level=1)
    new_df.columns = [f'{x}_{y}' for x, y in new_df.columns]
    new_df = new_df.reset_index()
    base_prob = new_df.iloc[:, :7]

    base_prob = base_prob.rename(columns={'id': 'strain', 'base_prob_1': 'prob.top', 'base_var_1': 'assign.top',
                                          'base_prob_2': 'prob.2', 'base_var_2': 'assign.2', 'base_prob_3': 'prob.3',
                                          'base_var_3': 'assign.3'})

    base_prob['assign.final'] = np.where(base_prob["prob.top"] < 0.25, "undetermined", base_prob["assign.top"])
    base_prob['assign.2'] = np.where(base_prob["prob.2"] == 0, 'NA', base_prob['assign.2'])
    base_prob['assign.3'] = np.where(base_prob["prob.3"] == 0, 'NA', base_prob['assign.3'])
    base_prob = base_prob.fillna('NA')

    if base_prob['assign.final'].str.contains('undetermined').any():
        dum_d2 = dum_d.loc[dum_d.index.isin(base_prob['strain'].loc[base_prob['assign.final'] == "undetermined"])]
        base_feat_lin = dum_d2.loc[:, dum_d2.columns.isin(model2.feature_names_in_)]
        base_pred_lin = model2.predict_proba(base_feat_lin)
        col = (model2.classes_).tolist()
        ind = list(base_feat_lin.index)
        base_pred_lin = pd.DataFrame(base_pred_lin, columns=col, index=ind)
        base_pred_lin['lin'] = base_pred_lin.apply(lambda x: list(zip(x.index[x > 0], x[x > 0])), axis=1)
        base_pred_lin['id'] = base_pred_lin.index
        lstsort = [(row.id, sorted(row.lin, key=lambda x: (-x[1], x[0]))) for row in base_pred_lin.itertuples()]
        base_prob_lin = pd.DataFrame(lstsort, columns=['id', 'base_lin_prob'])

        new_df = base_prob_lin.explode('base_lin_prob').reset_index(drop=True)
        new_df[['base_lin', 'base_prob_lin']] = pd.DataFrame(new_df['base_lin_prob'].tolist(), index=new_df.index)
        new_df = new_df[['id', 'base_lin', 'base_prob_lin']]
        new_df['idx'] = new_df.groupby('id').cumcount() + 1
        new_df = new_df.pivot_table(index=['id'], columns='idx', values=['base_lin', 'base_prob_lin'], aggfunc='first')
        new_df = new_df.sort_index(axis=1, level=1)
        new_df.columns = [f'{x}_{y}' for x, y in new_df.columns]
        new_df = new_df.reset_index()
        base_prob_lin = new_df.iloc[:, :7]

        base_prob_lin = base_prob_lin.rename(columns={'id': 'strain', 'base_prob_lin_1': 'prob.top', 'base_lin_1': 'assign.top',
                                                      'base_prob_lin_2': 'prob.2', 'base_lin_2': 'assign.2',
                                                      'base_prob_lin_3': 'prob.3',
                                                      'base_lin_3': 'assign.3'})

        base_prob_lin['assign.final'] = np.where(base_prob_lin["prob.top"] < 0.25, "undetermined", base_prob_lin["assign.top"])
        base_prob_lin['assign.2'] = np.where(base_prob_lin["prob.2"] == 0, 'NA', base_prob_lin['assign.2'])
        base_prob_lin['assign.3'] = np.where(base_prob_lin["prob.3"] == 0, 'NA', base_prob_lin['assign.3'])
        base_prob_lin = base_prob_lin.fillna('NA')
        base_prob_lin['assign.final2'] = np.where(base_prob_lin["assign.final"] == "undetermined", "undetermined", base_prob_lin["assign.final"] + "-unclassified")

        base_prob_lin_merge = base_prob_lin[['strain', 'assign.final2']]
        base_prob_lin_merge = base_prob_lin_merge.rename(columns={'assign.final2': 'assign.final'})

        base_prob_all = pd.merge(base_prob, base_prob_lin_merge, on=['strain'], how='outer')
        base_prob_all['assign.final'] = base_prob_all['assign.final_y']
        base_prob_all.loc[base_prob_all['assign.final'].isna(), 'assign.final'] = base_prob_all['assign.final_x']
        base_prob_all = base_prob_all.drop(['assign.final_x', 'assign.final_y'], axis=1)
    else:
        base_prob_all = base_prob

    final = base_prob_all[['strain', 'assign.final', 'assign.top', 'prob.top', 'assign.2', 'prob.2', 'assign.3', 'prob.3']]

    outfile = args.out
    final.to_csv(outfile, sep=',', header=True, index=False)