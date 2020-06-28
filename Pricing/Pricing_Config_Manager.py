import json


class Pricing_Config_Manager:
    def __init__(self, id):
        self.id = id
        with open('Pricing/configs/pricing_env.json') as json_file:
            data = json.load(json_file)
        campaign = data["campaigns"][id]

        # Features
        self.features = campaign["features"]
        self.feature_space = self.get_feature_space()

        # Environment
        self.categories = [tuple(self.features[f][c[f]] for f in self.features) for c in campaign["categories"]]
        self.prices = campaign["prices"]
        self.probabilities = campaign["probabilities"]


    def get_feature_space(self):
        def get_feature_space_rec(features, feature_list, values):
            if len(feature_list) == 0:
                feature_space.append(tuple(values))
            else:
                f = feature_list[0]
                for v in features[f]:
                    get_feature_space_rec(features, feature_list[1:], values+[v])

        feature_space = []
        features_list = list(self.features.keys())
        get_feature_space_rec(self.features, features_list, [])

        return feature_space

    def get_indexed_categories(self):
        return {i: c for i, c in enumerate(self.categories)}