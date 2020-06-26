from .feature import get_features

def load_data(trivias_list):

    all_df = get_features(trivias_list)

    y = all_df['norm_hee']
    X = all_df.drop('norm_hee', axis=1)

    return X, y
    