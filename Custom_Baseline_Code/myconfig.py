my_config = {
    "program": "main.py",
    "method": "grid",
    "name": "Sweep",
    "parameters":
    {
        "random_seed":
            {"value": 21},
        "dataset_path":
            {"value": '/opt/ml/segmentation/input/data'},
        "train_path":
            {"value": '/opt/ml/segmentation/input/data/train.json'},
        "val_path":
            {"value": '/opt/ml/segmentation/input/data/val.json'},
        "test_path":
            {"value": '/opt/ml/segmentation/input/data/test.json'},
        "dataset":
            {"value": 'CustomDataLoader'},
        "category_names":
            {"value": ('Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')},
        "augmentation":
            {"values": ['train_transform_1', 'train_transform']},
            #{"value": 'train_transform'},
        # "resize":
        #     {"value": (256, 192)},
        "optimizer":
            {"values": ["adam","adamw"]},
            #{"value": "adam"},
        "model":
            {"value": 'fcn_resnet50'},
        "batch_size":
            {"value": 16},
        "lr":
            {"value": 0.0001},
        "lr_decay_step":
            {"value": 10},
        "criterion":
            {"value": 'cross_entropy'}, #, 'focal', 'label_smoothing', 'F1Loss']},
        "num_epochs":
            {"value": 5},
        "scheduler":
            #{"values": ['steplr', 'CosineAnnealingLR']},
            {"value": 'steplr'},
        "saved_dir":
            {"value": './saved'},
        "val_every":
            {"value": 1},
        "name":
            {"value": 'sweep'},
    }
}