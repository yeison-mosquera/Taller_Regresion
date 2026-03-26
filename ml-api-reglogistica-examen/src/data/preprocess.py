def preprocess(df):
    X = df[['Horas']]
    y = (df['Notas'] >= 60).astype(int)
    return X, y