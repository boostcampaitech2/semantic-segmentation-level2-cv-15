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
        # "augmentation":
        #     {"value": 'my_transform'},
        # "resize":
        #     {"value": (256, 192)},
        "optimizer":
            {"values": ['adam']},
        "model":
            {"values": ['fcn_resnet50']},
        "batch_size":
            {"values": [16]},
        "lr":
            {"values": [0.00005]},
        "lr_decay_step":
            {"value": 10},
        "milestones":
            {"values":[30,80]},
        "T_max":
            {"value":50},
        "eta_min":
            {"value":0},
        "criterion":
            {"values": ['cross_entropy']}, #, 'focal', 'label_smoothing', 'F1Loss']},
        "num_epochs":
            {"value": 40},
        "scheduler":
            {"value": 'multisteplr'},
        "saved_dir":
            {"value": './saved'},
        "val_every":
            {"value": 1},
        "name":
            {"value": 'sweep'},
    }
}