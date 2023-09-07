import logger
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OrdinalEncoder


class ModelPreprocessing:

    @staticmethod
    def fit_cat_encoder(data):
        """
        This functions accepts a dataframe with all categorical features as inout and returns a fitted encoder for transforming similar dataframe
        The technique or method and the features to be encoded is defined within the method.

        Parameter
        data : a dataframe whose data needs to be transformed
        """

        try:
            zoning_cat = ['C (all)', 'RH', 'FV', 'RM', 'RL']
            street_cat = ['Grvl', 'Pave']
            alley_cat = ['Pave', 'Grvl', 'No alley']
            lot_shape_cat = ['IR3', 'IR2', 'IR1', 'Reg']
            land_contour_cat = ['Low', 'HLS', 'Bnk', 'Lvl']
            utilities_cat = ['ELO', 'NoSeWa', 'NoSewr', 'AllPub']
            lot_config_cat = ['FR3', 'FR2', 'CulDSac', 'Corner', 'Inside']
            land_slope_cat = ['Sev', 'Mod', 'Gtl']
            neighborhood_cat = ['Blueste', 'Veenker', 'NPkVill', 'BrDale', 'Blmngtn', 'MeadowV', 'StoneBr', 'SWISU', 'ClearCr', 'NoRidge', 'Timber', 'IDOTRR', 'Mitchel', 'BrkSide',
                                'Crawfor', 'SawyerW', 'Sawyer', 'NWAmes', 'NridgHt', 'Gilbert', 'Somerst', 'Edwards', 'OldTown', 'CollgCr', 'NAmes']
            condition1_cat = ['RRNe', 'RRNn', 'PosA', 'RRAe', 'PosN', 'RRAn', 'Artery', 'Feedr', 'Norm']
            condition2_cat = ['RRNe', 'RRNn', 'PosA', 'RRAe', 'PosN', 'RRAn', 'Artery', 'Feedr', 'Norm']
            bldg_type_cat = ['2fmCon', 'Duplex', 'Twnhs', 'TwnhsE', '1Fam']
            house_style_cat = ['2.5Fin', '2.5Unf', '1.5Unf', 'SFoyer', 'SLvl', '1.5Fin', '2Story', '1Story']
            roof_style_cat = ['Shed', 'Mansard', 'Gambrel', 'Flat', 'Hip', 'Gable']
            roof_matl_cat = ['ClyTile', 'Roll', 'Membran', 'Metal', 'WdShake', 'WdShngl', 'Tar&Grv', 'CompShg']
            exterior_1st = ['CBlock', 'Other', 'PreCast', 'AsphShn', 'ImStucc', 'BrkComm', 'Stone', 'AsbShng', 'Stucco', 'WdShing', 'BrkFace', 'CemntBd',
                            'Plywood', 'MetalSd', 'Wd Sdng', 'HdBoard', 'VinylSd']
            exterior_2nd = ['CBlock', 'Other', 'PreCast', 'AsphShn', 'ImStucc', 'BrkComm', 'Stone', 'AsbShng', 'Stucco', 'WdShing', 'BrkFace', 'CemntBd',
                            'Plywood', 'MetalSd', 'Wd Sdng', 'HdBoard', 'VinylSd']
            mas_vnr_type_cat = ['No walls', 'BrkCmn', 'Stone', 'BrkFace', 'None']
            exter_qual_cat = ['Po', 'Fa', 'TA', 'Ex', 'Gd']
            exter_cond_cat = ['Po', 'Fa', 'Gd', 'Ex', 'TA']
            foundation_cat = ['Wood', 'Stone', 'Slab', 'BrkTil', 'CBlock', 'PConc']
            bsmt_qual_cat = ['No Basement', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
            bsmt_cond_cat = ['No Basement', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
            bsmt_exposure_cat = ['No Basement', 'No', 'Mn', 'Av', 'Gd']
            bsmt_fin_type1_cat = ['No Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
            bsmt_fin_type2_cat = ['No Basement', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
            heating_cat = ['OthW', 'Floor', 'Wall', 'Grav', 'GasW', 'GasA']
            heating_qc_cat = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
            central_air_cat = ['N', 'Y']
            electrical_cat = ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']
            kitchen_qual_cat = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
            functional_cat = ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min1', 'Min2', 'Typ']
            fireplace_qu_cat = ['No Fireplace', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
            garage_type_cat = ['2Types', 'CarPort', 'Basment', 'BuiltIn', 'No Garage', 'Detchd', 'Attchd']
            garage_finish_cat = ['No Garage', 'Unf', 'RFn', 'Fin']
            garage_qual_cat = ['No Garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
            garage_cond_cat = ['No Garage', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
            paved_drive_cat = ['P', 'N', 'Y']
            pool_qc_cat = ['No Pool', 'Fa', 'TA', 'Gd', 'Ex']
            fence_cat = ['MnWw', 'GdWo', 'GdPrv', 'MnPrv', 'No Fence']
            misc_feature_cat = ['Elev', 'TenC', 'Gar2', 'Othr', 'Shed', 'No MiscFeature']
            sale_type_cat = ['VWS', 'Con', 'Oth', 'CWD', 'ConLw', 'ConLI', 'ConLD', 'COD', 'New', 'WD']
            sale_condition_cat = ['AdjLand', 'Alloca', 'Family', 'Abnorml', 'Partial', 'Normal']

            cat_features = {'MSZoning': zoning_cat, 'Street': street_cat, 'Alley': alley_cat, 'LotShape': lot_shape_cat, 'LandContour': land_contour_cat, 'Utilities': utilities_cat,
                            'LotConfig': lot_config_cat, 'LandSlope': land_slope_cat, 'Neighborhood': neighborhood_cat, 'Condition1': condition1_cat, 'Condition2': condition2_cat,
                            'BldgType': bldg_type_cat, 'HouseStyle': house_style_cat, 'RoofStyle': roof_style_cat, 'RoofMatl': roof_matl_cat, 'Exterior1st': exterior_1st,
                            'Exterior2nd': exterior_2nd, 'MasVnrType': mas_vnr_type_cat, 'ExterQual': exter_qual_cat, 'ExterCond': exter_cond_cat, 'Foundation': foundation_cat,
                            'BsmtQual': bsmt_qual_cat, 'BsmtCond': bsmt_cond_cat, 'BsmtExposure': bsmt_exposure_cat, 'BsmtFinType1': bsmt_fin_type1_cat,
                            'BsmtFinType2': bsmt_fin_type2_cat, 'Heating': heating_cat, 'HeatingQC': heating_qc_cat, 'CentralAir': central_air_cat, 'Electrical': electrical_cat,
                            'KitchenQual': kitchen_qual_cat, 'Functional': functional_cat, 'FireplaceQu': fireplace_qu_cat, 'GarageType': garage_type_cat,
                            'GarageFinish': garage_finish_cat, 'GarageQual': garage_qual_cat, 'GarageCond': garage_cond_cat, 'PavedDrive': paved_drive_cat, 'PoolQC': pool_qc_cat,
                            'Fence': fence_cat, 'MiscFeature': misc_feature_cat, 'SaleType': sale_type_cat, 'SaleCondition': sale_condition_cat}

            cat_encoder = DataFrameMapper([([feature], OrdinalEncoder(categories=[feature_cat])) for feature, feature_cat in cat_features.items()], df_out=True)
            encoder = cat_encoder.fit(data)

            logger.logging.info('Categorical features encoder fitted successfully.')
            return encoder

        except Exception as x:
            print('Error in fitting categorical encoder. Check logs.')
            logger.logging.exception('Error while fitting categorical encoder: ' + str(x))
            return None

    @staticmethod
    def plot_feature_importance(features_list, fitted_model, algo_name='Tree Based Model', plot_size=(20, 30), font_scale=2):
        """
        This functions accepts list of features, fitted model, algorithm used to fit the model and plot size and draws a plot of feature importance for tree based algorithms

        Parameter
        features_list: list - A list of feature names
        fitted_model: Fitted model for which features importance plot is to be plotted
        algo_name: str - Name of the algorith used to train/fit the model
        plot_size: (x, y) - Dimension of the resultant plot
        font_scale: int, default 2 - To adjust font size in the plot
        """

        try:
            # Calculating feature importance
            feature_imp = pd.DataFrame({'feature_name': features_list, 'feature_importance': fitted_model.feature_importances_})
            feature_imp.sort_values(by='feature_importance', ascending=False, inplace=True)

            # Plotting feature importance plot
            plt.figure(figsize=plot_size)
            sns.set(font_scale=font_scale)
            sns.barplot(x=feature_imp['feature_importance'], y=feature_imp['feature_name'])
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title(algo_name + "'s Feature Importance")
            plt.show()

            logger.logging.info('Feature Importance plot plotted successfully.')

        except Exception as x:
            print('Error in plotting feature importance plot. Check logs.')
            logger.logging.exception('Error in plotting plot for feature importance:' + str(x))

    @staticmethod
    def remove_insignificant_feature(data, fitted_model, threshold):
        """
        This function accepts dataframe used to train the model, fitted model, threshold value of feature importance.
        The function removes feature from the dataframe with less importance than the threshold and returns that dataframe.

        Parameter
        data: df - data used for training the model
        fitted_model: fitted model for which insignificant feature need to be removed
        threshold: float - threshold value to filter features
        """

        try:
            data_frame = pd.DataFrame({'feature': list(data), 'Feature_imp': fitted_model.feature_importances_})

            # feature imp less than threshold value
            feat = []
            for row in range(data_frame.shape[0]):
                if data_frame.iloc[row, 1] < threshold:
                    feat.append(data_frame.iloc[row, 0])

            data_significant_feat = data.drop(feat, axis=1)

            logger.logging.info('Insignificant featured removed successfully.')
            return data_significant_feat

        except Exception as x:
            print('Error while removing insignificant feature from data. Check logs.')
            logger.logging.exception('Error while removing insignificant features:' + str(x))

    @staticmethod
    def continuous_features_to_bins(data, continuous_features):
        """
        This function accepts the dataframe and transforms the continuous features into different bins ranges and return the modified dataframe.

        Parameter
        data: df - original dataframe
        continuous_feature: list - list of continuous features which needs to be transformed
        """

        try:
            for features in continuous_features:

                if features == 'LotFrontage':
                    data['LotFrontage'] = pd.cut(x=data['LotFrontage'], bins=range(20, 321, 10), labels=range(0, 30))

                elif features == 'LotArea':
                    data['LotArea'] = pd.cut(x=data['LotArea'], bins=range(1000, 215501, 500), labels=range(0, 429))

                elif features == 'YearBuilt':
                    data['YearBuilt'] = pd.cut(x=data['YearBuilt'], bins=range(1870, 2011, 5), labels=range(0, 28))

                elif features == 'YearRemodAdd':
                    data['YearRemodAdd'] = pd.cut(x=data['YearRemodAdd'], bins=range(1950, 2016, 5), right=False, labels=range(0, 13))

                elif features == 'MasVnrArea':
                    data['MasVnrArea'] = pd.cut(x=data['MasVnrArea'], bins=range(0, 1651, 50), right=False, labels=range(0, 33))

                elif features == 'BsmtFinSF1':
                    data['BsmtFinSF1'] = pd.cut(x=data['BsmtFinSF1'], bins=range(0, 6001, 500), right=False, labels=range(0, 12))

                elif features == 'BsmtFinSF2':
                    data['BsmtFinSF2'] = pd.cut(x=data['BsmtFinSF2'], bins=range(0, 1601, 50), right=False, labels=range(0, 32))

                elif features == 'BsmtUnfSF':
                    data['BsmtUnfSF'] = pd.cut(x=data['BsmtUnfSF'], bins=range(0, 2501, 50), right=False, labels=range(0, 50))

                elif features == 'TotalBsmtSF':
                    data['TotalBsmtSF'] = pd.cut(x=data['TotalBsmtSF'], bins=range(0, 6501, 250), right=False, labels=range(0, 26))

                elif features == '1stFlrSF':
                    data['1stFlrSF'] = pd.cut(x=data['1stFlrSF'], bins=range(250, 5501, 250), labels=range(0, 21))

                elif features == '2ndFlrSF':
                    data['2ndFlrSF'] = pd.cut(x=data['2ndFlrSF'], bins=range(0, 2201, 50), right=False, labels=range(0, 44))

                elif features == 'LowQualFinSF':
                    data['LowQualFinSF'] = pd.cut(x=data['LowQualFinSF'], bins=range(0, 1101, 50), right=False, labels=range(0, 22))

                elif features == 'GrLivArea':
                    data['GrLivArea'] = pd.cut(x=data['GrLivArea'], bins=range(250, 6001, 250), labels=range(0, 23))

                elif features == 'GarageYrBlt':
                    data['GarageYrBlt'] = pd.cut(x=data['GarageYrBlt'], bins=range(1900, 2011, 5), labels=range(0, 22))
                    data['GarageYrBlt'].fillna(0, inplace=True)

                elif features == 'GarageArea':
                    data['GarageArea'] = pd.cut(x=data['GarageArea'], bins=range(0, 1501, 50), right=False, labels=range(0, 30))

                elif features == 'WoodDeckSF':
                    data['WoodDeckSF'] = pd.cut(x=data['WoodDeckSF'], bins=range(0, 1501, 50), right=False, labels=range(0, 30))

                elif features == 'OpenPorchSF':
                    data['OpenPorchSF'] = pd.cut(x=data['OpenPorchSF'], bins=range(0, 801, 50), right=False, labels=range(0, 16))

                elif features == 'EnclosedPorch':
                    data['EnclosedPorch'] = pd.cut(x=data['EnclosedPorch'], bins=range(0, 1001, 50), right=False, labels=range(0, 20))

                elif features == '3SsnPorch':
                    data['3SsnPorch'] = pd.cut(x=data['3SsnPorch'], bins=range(0, 601, 50), right=False, labels=range(0, 12))

                elif features == 'ScreenPorch':
                    data['ScreenPorch'] = pd.cut(x=data['ScreenPorch'], bins=range(0, 601, 50), right=False, labels=range(0, 12))

                elif features == 'PoolArea':
                    data['PoolArea'] = pd.cut(x=data['PoolArea'], bins=range(0, 1001, 50), right=False, labels=range(0, 20))

                elif features == 'MiscVal':
                    data['MiscVal'] = pd.cut(x=data['MiscVal'], bins=range(0, 20001, 250), right=False, labels=range(0, 80))

                elif features == 'YrSold':
                    data['YrSold'] = pd.cut(x=data['YrSold'], bins=range(2005, 2011, 1), labels=range(0, 5))

            logger.logging.info('All the features are transformed into bins.')
            return data

        except Exception as x:
            print('Error while transforming the feature. Check logs.')
            logger.logging.exception('Error while transforming features: ' + str(x))
            return None
