class DefaultConfigs(object):
    def __init__(self, model_name):
        self.model_name = model_name

    #1.string parameters
    train_data = "/home/sdhm/Projects/gpd2/models/new/15channels/1objects/train.h5"
    test_data = "/home/sdhm/Projects/gpd2/models/new/15channels/1objects/test.h5"
    val_data = "/home/sdhm/Projects/gpd2/models/new/15channels/1objects/test.h5"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "0"

    #2.numeric parameters
    fold = 0
    epochs = 30
    batch_size = 64
    img_height = 60
    img_weight = 60
    img_channels = 15
    num_classes = 2
    lr = 0.001
    weight_decay = 0.0005
    seed = 888


config = DefaultConfigs("lenet")
