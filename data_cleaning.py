def impute_categorical_features(dataframe):
    categorical_features = dataframe.select_dtypes(include='object').columns.to_list()
    print(categorical_features)
