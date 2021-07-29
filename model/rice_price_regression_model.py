'''
===============================================================================
# Rice price prediction model

Current:
     Evaluate the quality levels of 6 growth periods, use the 6 features to evaluate the summary quality.

Next step:
     CNN-LSTM to predict rice price and quality based on 6 images during growth period.
===============================================================================
'''

from .rice_health_model import RiceHealthModel
import statistics

class RiceQualityValidator():
    def __init__(self, user_id, crops_type, farm_area):
        """Initialize the parameters for TDNN
        Args:
          _uid: user id
          crops_type: category of crops
          farm_area: farm region
        """
        self._uid = user_id
        self._crops_cate = crops_type
        self._farm_area = farm_area
        # img_list contains 6 images of different growth periods,
        # currently we use 6 static images
        self._img_list = ['../data/ricegrowthimages/1.jpg', '../data/ricegrowthimages/2.jpg', '../data/ricegrowthimages/3.jpg', \
                          '../data/ricegrowthimages/4.jpg', '../data/ricegrowthimages/5.jpg', '../data/ricegrowthimages/6.jpg']

        # static price list
        prices = [2.73, 2.85, 2.97, 3.09, 3.21]

    def train(self):
        "Train CNN-LSTM model"
        pass

    def load_data(self):
        "Prepare data for CNN-LSTM"
        pass

    def predict_by_CNN_LSTM(self):
        pass

    def predict(self):
        model = RiceHealthModel()
        result = []
        for img in self._img_list:
            result.append(model.predict(img))
        mean_score = statistics.mean([item['score'] for item in result])
        level = ''
        if mean_score >= 90:
            level = 'A'
        elif mean_score >= 80:
            level = 'B'
        elif mean_score >= 70:
            level = 'C'
        elif mean_score >= 60:
            level = 'D'
        else:
            level = 'E'

        return {"score": mean_score, "level": level}















