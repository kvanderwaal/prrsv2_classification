import pandas as pd
import numpy as np
from Bio import SeqIO
import skops.io as sio
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimating PRRSV-2 phylogenetic variant using random forest')
    parser.add_argument('-s', '--seqali', type=str, required=True, help="PRRSV-2 multiple sequence alignment with length of 603 nt")
    parser.add_argument('-m', '--model', type=str, required=True, help="Trained random forest model")
    parser.add_argument('-o', '--out', type=argparse.FileType('w'), required=True, help="Name or full path of classification report in .csv format")
    args = parser.parse_args()

    model = sio.load(args.model, trusted=True)

    with open(args.seqali) as fp:
      records = [{'name': str(record.description),
                  'sequence': str(record.seq)} for record in SeqIO.parse(fp,"fasta")]

    d = pd.DataFrame.from_records(records)
    d['sequence'] = d['sequence'].apply(lambda x: x.lower())
    d = d.set_index('name')
    d = d['sequence'].apply(lambda x: pd.Series(list(x)))
    d = d.rename(columns={x:y for x,y in zip(d.columns,range(1,len(d.columns)+1))})
    d = d.add_prefix('p.')
    d[~d.isin(['a', 't', 'c', 'g', '-'])] = '-'
    d = d.replace('-', np.nan)

    for column in d.columns:
        d[column].fillna(d[column].mode()[0], inplace=True)

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

    base_feat = dum_d.loc[:, dum_d.columns.isin(model.feature_names_in_)]
    base_pred = model.predict_proba(base_feat)
    col = (model.classes_).tolist()
    ind = list(base_feat.index)
    base_pred = pd.DataFrame(base_pred, columns=col, index=ind)
    base_pred['var'] = base_pred.apply(lambda x: list(zip(x.index[x > 0], x[x > 0])), axis=1)
    base_pred['id'] = base_pred.index
    lstsort = [(row.id, sorted(row.var,key=lambda x:(-x[1], x[0]))) for row in base_pred.itertuples()]
    base_prob = pd.DataFrame(lstsort, columns=['id', 'base_var_prob'])

    new_df = base_prob.explode('base_var_prob').reset_index(drop=True)
    new_df[['base_var', 'base_prob']] = pd.DataFrame(new_df['base_var_prob'].tolist(), index=new_df.index)
    new_df = new_df[['id', 'base_var', 'base_prob']]
    new_df['idx'] = new_df.groupby('id').cumcount() + 1
    new_df = new_df.pivot_table(index=['id'], columns='idx', values=['base_var', 'base_prob'], aggfunc='first')
    new_df = new_df.sort_index(axis=1, level=1)
    new_df.columns = [f'{x}_{y}' for x, y in new_df.columns]
    new_df = new_df.reset_index()
    base_prob = new_df.iloc[:, : 7]

    base_prob = base_prob.rename(columns={'id': 'strain', 'base_prob_1': 'prob.top', 'base_var_1': 'assign.top',
                                          'base_prob_2': 'prob.2', 'base_var_2': 'assign.2', 'base_prob_3': 'prob.3',
                                          'base_var_3': 'assign.3'})

    base_prob['assign.final'] = np.where(base_prob["prob.top"] < 0.25, "undetermined", base_prob["assign.top"])

    final = base_prob[['strain', 'assign.final', 'assign.top', 'prob.top', 'assign.2', 'prob.2', 'assign.3', 'prob.3']]

    outfile = args.out

    final.to_csv(outfile, sep=',', header=True, index=False)