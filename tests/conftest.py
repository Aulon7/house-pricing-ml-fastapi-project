import pandas as pd
import pytest

"""Pytest fixtures (sample) for the Kaggle Ames Housing dataset."""
@pytest.fixture
def sample_houses():
    return pd.DataFrame(
        {
            "Id": [1, 2, 3, 4],
            "MSSubClass": [20, 60, 70, 120],
            "Neighborhood": ["NAmes", "CollgCr", "NewArea", "NewArea"],
            "Alley": [None, "Grvl", None, "Pave"],
            "GarageType": ["Attchd", "Attchd", "Detchd", None],
            "Electrical": [None, "SBrkr", "FuseA", "SBrkr"],
            "LotFrontage": [None, 80.0, 70.0, None],
            "TotalBsmtSF": [800, 900, 700, 600],
            "1stFlrSF": [900, 950, 850, 800],
            "2ndFlrSF": [200, 250, 0, 300],
            "FullBath": [2, 1, 1, 2],
            "HalfBath": [1, 0, 1, 0],
            "BsmtFullBath": [1, 0, 1, 0],
            "BsmtHalfBath": [0, 1, 0, 0],
            "OpenPorchSF": [20, 30, 0, 10],
            "EnclosedPorch": [0, 0, 15, 0],
            "3SsnPorch": [0, 0, 0, 5],
            "ScreenPorch": [5, 0, 0, 0],
            "WoodDeckSF": [10, 20, 0, 30],
            "YearBuilt": [2000, 2005, 1990, 2010],
            "YearRemodAdd": [2005, 2005, 1990, 2015],
            "YrSold": [2010, 2010, 2010, 2016],
            "GarageArea": [400, 500, 300, 0],
            "Fireplaces": [1, 0, 2, 0],
            "SalePrice": [200000, 220000, 180000, 250000],
        }
    )