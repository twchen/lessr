from pathlib import Path
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument(
    '-d',
    '--dataset',
    choices=['diginetica', 'gowalla', 'lastfm'],
    required=True,
    help='the dataset name',
)
required.add_argument(
    '-f',
    '--filepath',
    required=True,
    help='the file for the dataset, i.e., "train-item-views.csv" for diginetica, '
    '"loc-gowalla_totalCheckins.txt" for gowalla, '
    '"userid-timestamp-artid-artname-traid-traname.tsv" for lastfm',
)
optional.add_argument(
    '-t',
    '--dataset-dir',
    default='datasets/{dataset}',
    help='the folder to save the preprocessed dataset',
)
parser._action_groups.append(optional)
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir.format(dataset=args.dataset))

if args.dataset == 'diginetica':
    from utils.data.preprocess import preprocess_diginetica

    preprocess_diginetica(dataset_dir, args.filepath)
else:
    from pandas import Timedelta
    from utils.data.preprocess import preprocess_gowalla_lastfm

    csv_file = args.filepath
    if args.dataset == 'gowalla':
        usecols = [0, 1, 4]
        interval = Timedelta(days=1)
        n = 30000
    else:
        usecols = [0, 1, 2]
        interval = Timedelta(hours=8)
        n = 40000
    preprocess_gowalla_lastfm(dataset_dir, csv_file, usecols, interval, n)
