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
        "train_transform":
            {"values": ['train_transform1']},
        "val_transform":
            {"value": 'val_transform'},
        "test_transform":
            {"value": 'test_transform'},
        "optimizer":
            {"value": 'adam'},
        "model":
            {"values": ['deeplabv3+']},
        "batch_size":
            {"value": 16},
        "lr":
            {"values": [0.00002]},
        "lr_decay_step":
            {"value": 10},
        "criterion":
            {"values": ['cross_entropy']},
        "num_epochs":
            {"value": 200},
        "scheduler":
            {"values": ['reducelronplateau']},
        "saved_dir":
            {"value": './saved'},
        "val_every":
            {"value": 1},
        "name":
            {"value": 'sweep'},
    }
}