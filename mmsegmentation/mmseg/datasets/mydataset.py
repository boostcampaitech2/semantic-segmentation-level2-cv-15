from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class MytrashDataset(CustomDataset):
    CLASSES = ('Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')
    PALETTE = [[0,0,0], [192,0,128], [192,192,255], [0,128,64], [128,0,0], [172,224,64], [244,64,60], [192,128,64], [255,200,224], [243,246,244], [128,0,192]]

    def __init__(self, **kwargs):
        super(MytrashDataset, self).__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)