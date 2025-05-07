import pandas as pd
from Bio import Phylo
from Bio import SeqIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Align import MultipleSeqAlignment
from Bio.Align import AlignInfo
from Bio.Seq import Seq
from io import StringIO
from ete3 import Tree
from functools import reduce
import numpy as np
import dendropy
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate updated data for model training')
    parser.add_argument('-n', '--name', type=str, required=True, help="Name of quarter in YYYY_MMM format for example 2024_Mar")
    parser.add_argument('-p', '--predata', type=str, required=True, help="Previous train data in CSV format")
    parser.add_argument('-d', '--data', type=str, required=True, help="Variant classification up to recent quarter (attr.roll) in CSV format")
    parser.add_argument('-t', '--tree', type=str, required=True, help="Input Newick Tree File (.contree file of recent quarter)")
    parser.add_argument('-f', '--fasta', type=str, required=True, help="Sequence alignment in Fasta format (.fasta file of recent quarter)")
    parser.add_argument('-o', '--outdir', type=str, required=True, help="Output directory for the results")
    args = parser.parse_args()

    #load old train data
    olddf = pd.read_csv(args.predata, sep=',', header=0)
    #load new attr roll data
    df = pd.read_csv(args.data, sep=',', header=0)
    df2 = df[['strain', 'date', 'lineage', 'variant.id.07.ac.rolling']]
    df2.columns = ['strain', 'date', 'lineage', 'variant']
    df2 = df2[df2["lineage"].str.startswith('1')] #keep only L1
    df2 = df2[df2['variant'].str.contains("unclassified") == False]
    df2 = df2.drop(df2[df2.variant == 'und'].index)

    #load new last three years IQTREE contree in different formats
    tf = Phylo.read(args.tree, 'newick')
    dentree = dendropy.Tree.get(path=args.tree, schema="newick")
    with open(args.tree, 'r') as file:
        tt = file.read()
    t = Tree(tt, format=1)
    taxalst = [(node.name) for node in tf.get_terminals(order='preorder')]
    l1list = df2["strain"].tolist()

    #prune tree to keep only L1
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    taxal1list = intersection(taxalst, l1list)
    t.prune(taxal1list, preserve_branch_length=True)
    t2 = t.write()
    t3 = Tree(t2, format=1)

    #label nodes
    edge = 0
    for node in t3.traverse():
        if not node.is_leaf():
            node.name = "NODE_%07d" % edge
        edge += 1

    #t3.write(format=1, outfile="".join([args.out, ".tree"])) #in case need node labeled tree for treetime

    #Datetime of Tob (observation time point)
    name = args.name
    Tob = pd.to_datetime(name, format="%Y_%b")

    ##recent year prevalence parameters
    df3 = df2[['strain', 'date']]
    df3 = df3[df3["strain"].isin(taxal1list)]
    #df3.to_csv("".join([args.out, "_date.tsv"]), sep='\t', header=True, index=False) #in case need date metadata for treetime

    df3['date'] = pd.to_datetime(df3['date'])
    mindate = Tob - pd.DateOffset(years=1)
    recentyr = df3['strain'][(df3['date'] > mindate) & (df3['date'] <= Tob)].tolist()

    #load last three years fasta alignment
    with open(args.fasta) as seqs:
        record_dict = SeqIO.to_dict(SeqIO.parse(seqs, 'fasta'))
        record_dict = {key.replace(':', '_'): value for key, value in record_dict.items()}
        result_records = [record_dict[id_] for id_ in taxal1list]

    #group by variant
    df4 = df2[df2["strain"].isin(taxal1list)]
    df4 = df4.groupby('variant')['strain'].apply(list)
    df4 = df4.to_frame().reset_index()

    #calculate consensus nucleotide and aa sequences
    def consensus(lst):
        variant_records = [record_dict[id_] for id_ in lst]
        alignment = MultipleSeqAlignment(variant_records)
        summary_align = AlignInfo.SummaryInfo(alignment)
        consensus = summary_align.dumb_consensus(threshold=0.3, ambiguous='N')
        return consensus

    prevRNA = consensus(recentyr)
    prevAA = Seq(prevRNA).translate(to_stop=True)
    prevecto1 = prevAA[32:64]
    prevecto2 = prevAA[100:109]
    prevepiA = prevAA[27:30]
    prevepiB = prevAA[37:45]
    prevepiC = prevAA[52:61]
    prevHVR1 = prevAA[32:36]
    prevHVR2 = prevAA[57:61]

    lstconRNA = [(row.variant, consensus(row.strain)) for row in df4.itertuples()]
    dconRNA = pd.DataFrame(lstconRNA)
    dconRNA.columns = ['variant', 'conRNA']

    lstconAA = [(row.variant, row.conRNA, Seq(row.conRNA).translate(to_stop=True)) for row in dconRNA.itertuples()]
    dconAA = pd.DataFrame(lstconAA)
    dconAA.columns = ['variant', 'conRNA', 'conAA']

    #calculate variant purity and get clade support
    t4 = t3.write(format=1)
    tl = Phylo.read(StringIO(t4), "newick")

    lstnode = [(row.variant, tl.common_ancestor(row.strain), tl.common_ancestor(row.strain).count_terminals(), len(row.strain)) for row in df4.itertuples()]
    dnode = pd.DataFrame(lstnode)
    dnode.columns = ['variant', 'Nodelabel', 'ntaxa', 'nvarseq']
    dnode['varpurity'] = dnode.nvarseq / dnode.ntaxa

    tb = Phylo.read(StringIO(t2), 'newick')
    for node in tb.get_nonterminals():
        node.name = str(node.confidence)

    lstboot = [(row.variant, tb.common_ancestor(row.strain)) for row in df4.itertuples()]
    dboot = pd.DataFrame(lstboot)
    dboot.columns = ['variant', 'CladeSupport']

    #calculate within variant mean genetic distance
    def gdist(lst):
        variant_records = [record_dict[id_] for id_ in lst]
        alignment = MultipleSeqAlignment(variant_records)
        calculator = DistanceCalculator('identity', )
        dismat = calculator.get_distance(alignment)
        arr = np.array(dismat)
        arr[arr == 0] = 'nan'
        gdist = np.nanmean(arr)
        return gdist

    dfg = df4[df4['strain'].map(len) > 1]

    lstgdist = [(row.variant, gdist(row.strain)) for row in dfg.itertuples()]
    dgdist = pd.DataFrame(lstgdist)
    dgdist.columns = ['variant', 'gdist']

    #calculate within variant patristic distance
    pdm = dentree.phylogenetic_distance_matrix()
    labels = [str(t.label).replace(' ', '_') for t in dentree.taxon_namespace]
    dm = []
    for i, taxon_i in enumerate(dentree.taxon_namespace):
        row = []
        for j, taxon_j in enumerate(dentree.taxon_namespace):
            dist = pdm.distance(taxon_i, taxon_j)
            row.append(dist)
        dm.append(row)
    dm = np.array(dm)

    list_of_pairs = [(row.variant,
                      [dm[labels.index(row.strain[p1]), labels.index(row.strain[p2])] for p1 in range(len(row.strain)) for
                       p2 in range(p1 + 1, len(row.strain))]) for row in dfg.itertuples()]
    pdist = pd.DataFrame(list_of_pairs)
    pdist.columns = ['variant', 'rawpdist']
    pdist["pdist"] = pdist["rawpdist"].apply(np.nanmean)
    dpdist = pdist[['variant', 'pdist']]

    #combine tree and alignment data
    dfs = [df4, dconAA, dboot, dnode, dgdist, dpdist]
    nodedata = reduce(lambda left, right: pd.merge(left, right, on=['variant'], how='outer'), dfs)

    #breakdown each important aa site
    nodedata['ecto1'] = nodedata['conAA'].str[32:64:1]  # https://doi.org/10.1016/j.virol.2012.08.026
    nodedata['ecto2'] = nodedata['conAA'].str[100:109:1]  # https://doi.org/10.1016/j.virol.2012.08.026
    nodedata['epiA'] = nodedata['conAA'].str[27:30:1]  # doi: 10.1128/JVI.76.9.4241-4250.2002, https://doi.org/10.1016/j.vetmic.2017.04.016
    nodedata['epiB'] = nodedata['conAA'].str[37:45:1]  # doi: 10.1128/JVI.76.9.4241-4250.2002
    nodedata['epiC'] = nodedata['conAA'].str[52:61:1]  # https://doi.org/10.1016/j.micpath.2019.10365
    nodedata['HVR1'] = nodedata['conAA'].str[32:36:1]  # doi: 10.1128/spectrum.02634-21
    nodedata['HVR2'] = nodedata['conAA'].str[57:61:1]  # doi: 10.1128/spectrum.02634-21

    #calculate distances to the recent year consensus sequences
    def hamming_distance(string1, string2):
        # Start with a distance of zero, and count up
        distance = 0
        # Loop over the indices of the string
        L = len(string1)
        for i in range(L):
            # Add 1 to the distance if these two characters are not equal
            if string1[i] != string2[i]:
                distance += 1
        # Return the final count of differences
        return distance / L

    lstdist = [(row.variant, hamming_distance(row.conAA, prevAA),
                hamming_distance(row.ecto1, prevecto1),
                hamming_distance(row.ecto2, prevecto2),
                hamming_distance(row.epiA, prevepiA),
                hamming_distance(row.epiB, prevepiB),
                hamming_distance(row.epiC, prevepiC),
                hamming_distance(row.HVR1, prevHVR1),
                hamming_distance(row.HVR2, prevHVR2)) for row in nodedata.itertuples()]
    ddist = pd.DataFrame(lstdist)
    ddist.columns = ['variant', 'dist_conAA', 'dist_ecto1', 'dist_ecto2', 'dist_epiA', 'dist_epiB', 'dist_epiC',
                     'dist_HVR1', 'dist_HVR2']
    nodedata = pd.merge(nodedata, ddist, how="left", on="variant")

    #calculate EWMA-based parameters
    df2['date'] = pd.to_datetime(df2['date'])
    def ewmavar(tlen, colname):
        dfma = df2[['strain', 'variant', 'date']][(df2['date'] <= Tob)]
        dfma = dfma.sort_values(by=['variant', 'date'], ascending=[True, True])
        dfma['nseq'] = 1
        grouper_M = dfma[['variant', 'date', 'nseq']].set_index('date').groupby([pd.Grouper(freq=tlen), 'variant'])
        dfma_M = grouper_M['nseq'].sum().unstack('variant').fillna(0)
        dfma_M = dfma_M.cumsum(axis=0)  # use if want cumulative nseq for ewma
        dfma_M_ewm = dfma_M.ewm(alpha=0.5).mean()
        dfma_M = dfma_M.tail(1).unstack().reset_index().rename(columns={0: 'Mnseq'})
        dfma_M_ewm = dfma_M_ewm.tail(1).unstack().reset_index().rename(columns={0: 'Mewma'})
        dfma_merged = reduce(lambda left, right: pd.merge(left, right, on=['variant'], how='left'),
                             [dfma_M[['variant', 'Mnseq']], dfma_M_ewm[['variant', 'Mewma']]])
        # Create a clean suffix for column naming (e.g., '3M' from '3ME')
        label_suffix = tlen.replace('ME', 'M')
        dfma_merged[colname] = (dfma_merged['Mnseq'] - dfma_merged['Mewma']) * 100 / dfma_merged['Mewma']
        dfma_merged = dfma_merged.rename(columns={'Mewma': f"{label_suffix}ewma"})
        return dfma_merged[['variant', colname, f"{label_suffix}ewma"]]

    dew3M = ewmavar('3M', '3Mewmareldif')
    dew6M = ewmavar('6M', '6Mewmareldif')
    dew12M = ewmavar('12M', '12Mewmareldif')
    dew24M = ewmavar('24M', '24Mewmareldif')
    dew36M = ewmavar('36M', '36Mewmareldif')
    dew_merged = reduce(lambda left, right: pd.merge(left, right, on=['variant'], how='left'),
                        [dew3M, dew6M, dew12M, dew24M, dew36M])
    columns = ['3Mewma', '6Mewma', '12Mewma', '24Mewma', '36Mewma']

    for i in range(len(columns) - 1):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            new_col_name = f"{col1.split('M')[0]}_{col2.split('M')[0]}madif"
            dew_merged[new_col_name] = (dew_merged[col1] - dew_merged[col2]) * 100 / dew_merged[col2]

    dew_merged.drop(['3Mewma', '6Mewma', '12Mewma', '24Mewma', '36Mewma'], axis=1, inplace=True)
    dfss_merged = reduce(lambda left, right: pd.merge(left, right, on=['variant'], how='left'), [nodedata, dew_merged])

    #get season and varsize
    def get_season(date):
        month = date.month
        if (month == 12) or (month == 1) or (month == 2):
            return 'season_Winter'
        elif (month >= 3) and (month <= 5):
            return 'season_Spring'
        elif (month >= 6) and (month <= 8):
            return 'season_Summer'
        else:
            return 'season_Fall'

    dfss_merged['Tob'] = Tob
    dfss_merged['Season'] = dfss_merged['Tob'].apply(get_season)
    for season in ['season_Winter', 'season_Spring', 'season_Summer', 'season_Fall']:
        dfss_merged[season] = 0

    for season in ['season_Winter', 'season_Spring', 'season_Summer', 'season_Fall']:
        dfss_merged.loc[dfss_merged['Season'] == season, season] = 1

    dfss_merged = dfss_merged.drop(columns=['Season'])
    dfss_merged['varsize'] = dfss_merged['nvarseq']

    #append new data to old data
    olddf['Tob'] = pd.to_datetime(olddf['Tob'])
    dfss_merged_aligned = dfss_merged.reindex(columns=olddf.columns)
    df_new_old = pd.concat([olddf, dfss_merged_aligned], ignore_index=True)
    df_new_old = df_new_old.drop(df_new_old[df_new_old.nvarseq < 5].index)

    #calculate CAGR12 for variants use to have NA
    def nvarcount(df, colname):
        df = df.groupby('variant')['strain'].apply(list)
        df = df.to_frame().reset_index()
        lstdf = [(row.variant, len(row.strain)) for row in df.itertuples()]
        df = pd.DataFrame(lstdf)
        df.columns = ['variant', colname]
        return df
    dfca = df_new_old[df_new_old['CAGR_12'].isna()]
    dfca2 = dfca[dfca['Tob'] <= Tob - pd.DateOffset(years=1)]
    dfca2 = dfca2.drop(['CAGR_12', 'CAGR12_Class'], axis=1)

    if dfca2.empty:
        df_new_old = df_new_old
        print("No new CAGR12 to be calculated")
    else:
        df5 = df2[(df2['date'] <= max(dfca2['Tob']))]
        df5 = df5[(df5['variant'].isin(dfca2['variant']))]
        df5 = nvarcount(df5, 'nvarseqall')
        dfca_merged = reduce(lambda left, right: pd.merge(left, right, on=['variant'], how='left'), [dfca2, df5])

        df6 = df2[(df2['date'] > max(dfca2['Tob'])) & (df2['date'] <= max(dfca2['Tob']) + pd.DateOffset(years=1))]
        df6 = df6[(df6['variant'].isin(dfca2['variant']))]
        df6 = nvarcount(df6, 'nvarseqF12')
        dfca_merged = reduce(lambda left, right: pd.merge(left, right, on=['variant'], how='left'), [dfca_merged, df6])
        dfca_merged.update(dfca_merged['nvarseqF12'].fillna(0))

        def calculate_cagr(df, last_col='nvarseqF12', previous_col='nvarseqall', years=1):
            cagr = ((((df[last_col] + df[previous_col]) / df[previous_col]) ** (1 / years)) - 1) * 100
            return cagr

        dfca_merged['CAGR_12'] = calculate_cagr(dfca_merged, last_col='nvarseqF12', previous_col='nvarseqall', years=1)
        def classify_cagr(cagr):
            if cagr <= 20:
                return "low"
            elif cagr > 20:
                return "high"

        dfca_merged['CAGR12_Class'] = dfca_merged['CAGR_12'].apply(classify_cagr)
        dfca_merged = dfca_merged[dfca_merged['CAGR12_Class'].notna()]

        allmerged = df_new_old.merge(dfca_merged[['Tob', 'variant', 'CAGR_12', 'CAGR12_Class']], on=['Tob', 'variant'], how='left', suffixes=('', '_new'))

        # Replace values in df_new_old with values from dfca_merged where matches are found
        df_new_old['CAGR_12'] = allmerged['CAGR_12_new'].combine_first(df_new_old['CAGR_12'])
        df_new_old['CAGR12_Class'] = allmerged['CAGR12_Class_new'].combine_first(df_new_old['CAGR12_Class'])

    outfile = os.path.join(args.outdir, f"{args.name}_traindat.csv")
    df_new_old.to_csv(outfile, sep=',', header=True, index=False, na_rep='NA')
