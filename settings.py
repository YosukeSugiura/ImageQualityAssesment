#
# Settings for SEGAN
#

class settings:

    def __init__(self):

        #   Precision Mode
        self.halfprec = True                        # 16bit or not

        #   Image settings
        self.size               = (64,64)           # Input Size (64 by 64)

        #   Training
        self.batch_size = 100                       # Batch size
        self.epoch      = 2000                      # Epoch
        self.learning_rate = 0.000001                # Learning Rate

        # Retrain
        self.retrain    = 0

        # Save path
        self.model_save_path    = 'params'          # Network model path
        self.model_save_cycle   = 100               # Epoch cycle for saving model (init:1)
        self.result_save_path   = 'result'          # Network model path

        # Data path
        self.train_data_path    = './data/image_train'     # Folder containing training image (train)
        self.test_data_path     = './data/image_train'     # Folder containing test image (test)

        # Pkl files
        self.pkl_path     = 'pkl'             # Folder of pkl files for train
