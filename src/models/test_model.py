import xgboost as xgb
# from xgboost import XGBClassifier
print(xgb.__file__)
from src.models.base_model import BaseModel

class XgboostModel(BaseModel):
    def __init__(self, scale_pos_weight: int = 577):
        self.scale_pos_weight = scale_pos_weight

        super().__init__(
            model=xgb.XGBClassifier(scale_pos_weight=self.scale_pos_weight)
        )
