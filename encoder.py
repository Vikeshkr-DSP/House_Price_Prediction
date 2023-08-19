import logger
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OrdinalEncoder


class EncodeFeatures:

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
            logger.logging.exception('Error while fitting categorical encoder: ' + str(x))
            return None
