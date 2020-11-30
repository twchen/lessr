import pandas as pd
import numpy as np


def get_session_id(df, interval):
    df_prev = df.shift()
    is_new_session = (df.userId != df_prev.userId) | (
        df.timestamp - df_prev.timestamp > interval
    )
    session_id = is_new_session.cumsum() - 1
    return session_id


def group_sessions(df, interval):
    sessionId = get_session_id(df, interval)
    df = df.assign(sessionId=sessionId)
    return df


def filter_short_sessions(df, min_len=2):
    session_len = df.groupby('sessionId', sort=False).size()
    long_sessions = session_len[session_len >= min_len].index
    df_long = df[df.sessionId.isin(long_sessions)]
    return df_long


def filter_infreq_items(df, min_support=5):
    item_support = df.groupby('itemId', sort=False).size()
    freq_items = item_support[item_support >= min_support].index
    df_freq = df[df.itemId.isin(freq_items)]
    return df_freq


def filter_until_all_long_and_freq(df, min_len=2, min_support=5):
    while True:
        df_long = filter_short_sessions(df, min_len)
        df_freq = filter_infreq_items(df_long, min_support)
        if len(df_freq) == len(df):
            break
        df = df_freq
    return df


def truncate_long_sessions(df, max_len=20, is_sorted=False):
    if not is_sorted:
        df = df.sort_values(['sessionId', 'timestamp'])
    itemIdx = df.groupby('sessionId').cumcount()
    df_t = df[itemIdx < max_len]
    return df_t


def update_id(df, field):
    labels = pd.factorize(df[field])[0]
    kwargs = {field: labels}
    df = df.assign(**kwargs)
    return df


def remove_immediate_repeats(df):
    df_prev = df.shift()
    is_not_repeat = (df.sessionId != df_prev.sessionId) | (df.itemId != df_prev.itemId)
    df_no_repeat = df[is_not_repeat]
    return df_no_repeat


def reorder_sessions_by_endtime(df):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    df_endtime = endtime.sort_values().reset_index()
    oid2nid = dict(zip(df_endtime.sessionId, df_endtime.index))
    sessionId_new = df.sessionId.map(oid2nid)
    df = df.assign(sessionId=sessionId_new)
    df = df.sort_values(['sessionId', 'timestamp'])
    return df


def keep_top_n_items(df, n):
    item_support = df.groupby('itemId', sort=False).size()
    top_items = item_support.nlargest(n).index
    df_top = df[df.itemId.isin(top_items)]
    return df_top


def split_by_time(df, timedelta):
    max_time = df.timestamp.max()
    end_time = df.groupby('sessionId').timestamp.max()
    split_time = max_time - timedelta
    train_sids = end_time[end_time < split_time].index
    df_train = df[df.sessionId.isin(train_sids)]
    df_test = df[~df.sessionId.isin(train_sids)]
    return df_train, df_test


def train_test_split(df, test_split=0.2):
    endtime = df.groupby('sessionId', sort=False).timestamp.max()
    endtime = endtime.sort_values()
    num_tests = int(len(endtime) * test_split)
    test_session_ids = endtime.index[-num_tests:]
    df_train = df[~df.sessionId.isin(test_session_ids)]
    df_test = df[df.sessionId.isin(test_session_ids)]
    return df_train, df_test


def save_sessions(df, filepath):
    df = reorder_sessions_by_endtime(df)
    sessions = df.groupby('sessionId').itemId.apply(lambda x: ','.join(map(str, x)))
    sessions.to_csv(filepath, sep='\t', header=False, index=False)


def save_dataset(dataset_dir, df_train, df_test):
    # filter items in test but not in train
    df_test = df_test[df_test.itemId.isin(df_train.itemId.unique())]
    df_test = filter_short_sessions(df_test)

    print(f'No. of Clicks: {len(df_train) + len(df_test)}')
    print(f'No. of Items: {df_train.itemId.nunique()}')

    # update itemId
    train_itemId_new, uniques = pd.factorize(df_train.itemId)
    df_train = df_train.assign(itemId=train_itemId_new)
    oid2nid = {oid: i for i, oid in enumerate(uniques)}
    test_itemId_new = df_test.itemId.map(oid2nid)
    df_test = df_test.assign(itemId=test_itemId_new)

    print(f'saving dataset to {dataset_dir}')
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_sessions(df_train, dataset_dir / 'train.txt')
    save_sessions(df_test, dataset_dir / 'test.txt')
    num_items = len(uniques)
    with open(dataset_dir / 'num_items.txt', 'w') as f:
        f.write(str(num_items))


def preprocess_diginetica(dataset_dir, csv_file):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        usecols=[0, 2, 3, 4],
        delimiter=';',
        parse_dates=['eventdate'],
        infer_datetime_format=True,
    )
    print('start preprocessing')
    # timeframe (time since the first query in a session, in milliseconds)
    df['timestamp'] = pd.to_timedelta(df.timeframe, unit='ms') + df.eventdate
    df = df.drop(['eventdate', 'timeframe'], 1)
    df = df.sort_values(['sessionId', 'timestamp'])
    df = filter_short_sessions(df)
    df = truncate_long_sessions(df, is_sorted=True)
    df = filter_infreq_items(df)
    df = filter_short_sessions(df)
    df_train, df_test = split_by_time(df, pd.Timedelta(days=7))
    save_dataset(dataset_dir, df_train, df_test)


def preprocess_gowalla_lastfm(dataset_dir, csv_file, usecols, interval, n):
    print(f'reading {csv_file}...')
    df = pd.read_csv(
        csv_file,
        sep='\t',
        header=None,
        names=['userId', 'timestamp', 'itemId'],
        usecols=usecols,
        parse_dates=['timestamp'],
        infer_datetime_format=True,
    )
    print('start preprocessing')
    df = df.dropna()
    df = update_id(df, 'userId')
    df = update_id(df, 'itemId')
    df = df.sort_values(['userId', 'timestamp'])

    df = group_sessions(df, interval)
    df = remove_immediate_repeats(df)
    df = truncate_long_sessions(df, is_sorted=True)
    df = keep_top_n_items(df, n)
    df = filter_until_all_long_and_freq(df)
    df_train, df_test = train_test_split(df, test_split=0.2)
    save_dataset(dataset_dir, df_train, df_test)
