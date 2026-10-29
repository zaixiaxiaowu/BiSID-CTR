from .DeepFM import DeepFM


class SIDFeat(DeepFM):
    """DAS-style SID features: user/item SIDs as extra categorical fields on DeepFM."""

    def __init__(self, feature_map, model_id="SIDFeat", **kwargs):
        super(SIDFeat, self).__init__(feature_map, model_id=model_id, **kwargs)
