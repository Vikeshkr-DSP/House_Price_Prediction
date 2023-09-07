import logger
import pandas as pd
from sklearn.model_selection import train_test_split


class PreProcessingData:

    @staticmethod
    def split_df(raw_data, dependent_column, test_ratio=0.2, stratify_column=None):
        """
        This function returns the input DF by splitting it into 4 dataframes. Splitting is done on the basis dependent column in the provided DF.
        The DF splits into train and validation, 80% to train and 20% to validation by default.

        Parameter
        raw_data : *arrays, DataFrame
        dependent_column: strings - Dependent column name.
        test_ratio : default 0.2, float >0 and <1 to specify percentage of Data into validation DF
        stratify_column : default None - Strings - Column name to stratify the DF split
        """
        try:
            if stratify_column is not None:
                train, test = train_test_split(raw_data, test_size=test_ratio, shuffle=True, random_state=10, stratify=raw_data[[stratify_column]])
                logger.logging.info('Data split using stratified sampling: ' + stratify_column)

            else:
                train, test = train_test_split(raw_data, test_size=test_ratio, shuffle=True, random_state=10)
                logger.logging.info('Data split without stratified sampling.')

            x_train = train.drop([dependent_column], axis=1)
            y_train = train[[dependent_column]]
            x_valid = test.drop([dependent_column], axis=1)
            y_valid = test[[dependent_column]]

            del train, test
            logger.logging.info('Dataframe split into train and validation successful.')
            return x_train, y_train, x_valid, y_valid

        except Exception as x:
            logger.logging.exception('Error in splitting data: ' + str(x))
            return None

    @staticmethod
    def missing_data_count(raw_data):
        """
        This function calculates and returns missing value count and percentage in a feature or column of the provided dataframe.

        Parameter
        raw_data: Dataframe to find missing value count
        """
        try:
            missing = (raw_data.isnull().sum()).sort_values(ascending=False)
            missing_percent = round((missing / (len(raw_data))) * 100, 2)
            result = pd.concat(objs=[missing, missing_percent], axis=1, keys=['Missing Count', 'Missing Percentage'])

            if len(result[result['Missing Count'] > 0]) > 0:
                print(result[result['Missing Count'] > 0])

            else:
                print('None of the columns have missing value.')

            del missing, missing_percent, result
            logger.logging.info('Missing value count and percentage calculated.')

        except Exception as x:
            logger.logging.exception('Error in calculating missing value count or percentage of the DF: ' + str(x))
