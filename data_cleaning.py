import logger
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class ProcessingMissingValue:

    @staticmethod
    def impute_categorical_features(dataframe, fitted_impute_nan_cat, fitted_impute_none_cat):
        """
        This function accepts a dataframe, fitted sklearn model to impute nan & null and imputes categorical features in the dataframe

        Parameter
        dataframe: Accepts dataframe as input
        impute_nan_cat: A fitted model to transform nan values
        impute_none_cat: A fitted model to transform None value
        """
        try:
            categorical_features = dataframe.select_dtypes(include='object').columns.to_list()

            # Features below are imputed as described in data_description file
            for feature in categorical_features:

                try:
                    if feature == 'PoolQC' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Pool', inplace=True)
                        logger.logging.info('{0} imputed with "No Pool" successfully.'.format(feature))

                    elif feature == 'MiscFeature' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No MiscFeature', inplace=True)
                        logger.logging.info('{0} imputed with "No MiscFeature" successfully.'.format(feature))

                    elif feature == 'Alley' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No alley', inplace=True)
                        logger.logging.info('{0} imputed with "No alley" successfully.'.format(feature))

                    elif feature == 'Fence' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Fence', inplace=True)
                        logger.logging.info('{0} imputed with "No Fence" successfully.'.format(feature))

                    elif feature == 'FireplaceQu' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Fireplace', inplace=True)
                        logger.logging.info('{0} imputed with "No Fireplace" successfully.'.format(feature))

                    elif feature == 'GarageType' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Garage', inplace=True)
                        logger.logging.info('{0} imputed with "No Garage" successfully.'.format(feature))

                    elif feature == 'GarageQual' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Garage', inplace=True)
                        logger.logging.info('{0} imputed with "No Garage" successfully.'.format(feature))

                    elif feature == 'GarageCond' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Garage', inplace=True)
                        logger.logging.info('{0} imputed with "No Garage" successfully.'.format(feature))

                    elif feature == 'GarageFinish' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Garage', inplace=True)
                        logger.logging.info('{0} imputed with "No Garage" successfully.'.format(feature))

                    elif feature == 'BsmtExposure' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Basement', inplace=True)
                        logger.logging.info('{0} imputed with "No Basement" successfully.'.format(feature))

                    elif feature == 'BsmtCond' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Basement', inplace=True)
                        logger.logging.info('{0} imputed with "No Basement" successfully.'.format(feature))

                    elif feature == 'BsmtFinType2' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Basement', inplace=True)
                        logger.logging.info('{0} imputed with "No Basement" successfully.'.format(feature))

                    elif feature == 'BsmtFinType1' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Basement', inplace=True)
                        logger.logging.info('{0} imputed with "No Basement" successfully.'.format(feature))

                    elif feature == 'BsmtQual' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No Basement', inplace=True)
                        logger.logging.info('{0} imputed with "No Basement" successfully.'.format(feature))

                    elif feature == 'MasVnrType' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna('No walls', inplace=True)
                        logger.logging.info('{0} imputed with "No walls" successfully.'.format(feature))

                except Exception as x:
                    logger.logging.exception('Error in imputing {0} : '.format(feature) + str(x))

            try:
                # Imputing missing values for features where nan means missing value
                nan_feature_impute = pd.DataFrame(fitted_impute_nan_cat.transform(dataframe[['MSZoning', 'Utilities', 'Functional', 'KitchenQual', 'Exterior2nd',
                                                                                             'Exterior1st', 'SaleType']]),
                                                  columns=['MSZoning', 'Utilities', 'Functional', 'KitchenQual', 'Exterior2nd', 'Exterior1st', 'SaleType'],
                                                  index=dataframe.index)

                dataframe[['MSZoning', 'Utilities', 'Functional', 'KitchenQual', 'Exterior2nd', 'Exterior1st', 'SaleType']] = nan_feature_impute

                del nan_feature_impute
                logger.logging.info('Missing values imputed successfully for MSZoning, Utilities, Functional, KitchenQual, Exterior2nd, Exterior1st and ' +
                                    'SaleType with most frequent category.')

            except Exception as x:
                logger.logging.exception('Error in imputing features having nan as null: ' + str(x))

            try:
                # Imputing missing values for features where None means missing value
                none_feature_impute = pd.DataFrame(fitted_impute_none_cat.transform(dataframe[['Electrical']]),
                                                   columns=['Electrical'],
                                                   index=dataframe.index)

                dataframe[['Electrical']] = none_feature_impute

                del none_feature_impute
                logger.logging.info('Missing values imputed successfully for Electrical with most frequent category.')

            except Exception as x:
                logger.logging.exception('Error in imputing features having None as null: ' + str(x))

        except Exception as x:
            logger.logging.exception('Error in imputing categorical features: ' + str(x))

    @staticmethod
    def fit_model_none_cat(df, none_features):
        """
        This function accepts dataframe and list of features where missing values (None) need to be imputed.
        Returns a fitted model of sklearn which can be used to impute same feature in dataframe.

        Parameter
        df: Dataframe
        none_features: List of categorical features to be imputed by most frequent categories with None as missing value
        """

        try:
            impute_feature_none = SimpleImputer(missing_values=None, strategy='most_frequent')
            impute_feature_none.fit(df[none_features])

            logger.logging.info('Sklearn model to impute None fitted successfully.')
            del df, none_features
            return impute_feature_none

        except Exception as x:
            logger.logging.exception('Error in fitting the model to impute None values: ' + str(x))
            return None

    @staticmethod
    def fit_model_nan_cat(df, nan_features):
        """
        This function accepts a dataframe and list of features where missing values (nan) need to be imputed
        Returns a fitted model of sklearn which can be used to impute same feature in dataframe

        Parameter
        df: Dataframe
        nan_features: List of categorical features to be imputed by most frequent categories with nan as missing value
        """

        try:
            impute_frequent_nan = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
            impute_frequent_nan.fit(df[nan_features])

            del df, nan_features
            logger.logging.info('Sklearn model to impute NaN fitted successfully.')
            return impute_frequent_nan

        except Exception as x:
            logger.logging.exception('Error in fitting the model to impute NaN values: ' + str(x))
            return None

    @staticmethod
    def impute_continuous_features(dataframe, fitted_impute_nan_cont):
        """
        This function accepts a dataframe and a fitted model to impute continuous features.

        Parameter
        dataframe: df
        fitted_impute_nan_cont: A fitted model to transform continuous missing values (NaN)
        """

        try:
            continuous_features = dataframe.select_dtypes(exclude='object').columns.to_list()

            # Features below are imputed with 0 based on description in data_description file
            for feature in continuous_features:
                try:

                    if feature == 'MasVnrArea' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna(0, inplace=True)
                        logger.logging.info('{0} imputed with 0 successfully.'.format(feature))

                    elif feature == 'BsmtHalfBath' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna(0, inplace=True)
                        logger.logging.info('{0} imputed with 0 successfully.'.format(feature))

                    elif feature == 'BsmtFullBath' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna(0, inplace=True)
                        logger.logging.info('{0} imputed with 0 successfully.'.format(feature))

                    elif feature == 'BsmtFinSF1' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna(0, inplace=True)
                        logger.logging.info('{0} imputed with 0 successfully.'.format(feature))

                    elif feature == 'BsmtFinSF2' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna(0, inplace=True)
                        logger.logging.info('{0} imputed with 0 successfully.'.format(feature))

                    elif feature == 'BsmtUnfSF' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna(0, inplace=True)
                        logger.logging.info('{0} imputed with 0 successfully.'.format(feature))

                    elif feature == 'TotalBsmtSF' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna(0, inplace=True)
                        logger.logging.info('{0} imputed with 0 successfully.'.format(feature))

                    elif feature == 'GarageYrBlt' and dataframe[feature].isnull().sum() > 0:
                        dataframe[feature].fillna(0000, inplace=True)
                        logger.logging.info('{0} imputed with 0 successfully.'.format(feature))

                except Exception as x:
                    logger.logging.exception('Error in imputing {0}.'.format(feature) + str(x))

            try:
                # Imputing ['GarageCars', 'GarageArea'] with their respective medians
                impute_nan_median_cont = pd.DataFrame(fitted_impute_nan_cont.transform(dataframe[['LotFrontage', 'GarageCars', 'GarageArea']]),
                                                      columns=['LotFrontage', 'GarageCars', 'GarageArea'],
                                                      index=dataframe.index)
                dataframe[['LotFrontage', 'GarageCars', 'GarageArea']] = impute_nan_median_cont

                del impute_nan_median_cont
                logger.logging.info('Missing values imputed successfully with medians for LotFrontage, GarageCars & GarageArea.')

            except Exception as x:
                logger.logging.exception('Error while imputing continuous features with medians: ' + str(x))

        except Exception as x:
            logger.logging.exception('Error in imputing continuous features: ' + str(x))

    @staticmethod
    def fit_model_nan_cont(df, features_list):
        """
        This function accepts a dataframe and list of features where missing values (nan) need to be imputed with median.
        Returns a fitted model of sklearn which can be used to impute same feature in dataframe.

        Parameter
        df: Dataframe
        features_list: List of categorical features to be imputed by median with nan as missing value
        """
        try:
            median_impute_fit = SimpleImputer(missing_values=np.NaN, strategy='median')
            median_impute_fit.fit(df[features_list])

            del df, features_list
            logger.logging.info('Sklearn model to impute missing continuous values with median trained successfully.')
            return median_impute_fit

        except Exception as x:
            logger.logging.exception('Error in fitting model to impute NaN with median: ' + str(x))
            return None
