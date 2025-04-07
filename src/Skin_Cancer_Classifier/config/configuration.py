from Skin_Cancer_Classifier.constants import *
from Skin_Cancer_Classifier.utils.common import read_yaml, create_directories
from Skin_Cancer_Classifier.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        root_dir = Path(config.root_dir)
        arch_name = self.params.ARCH_NAME
        base_model_path = root_dir / f"model_{arch_name}.pth"

        prepare_base_model_config = PrepareBaseModelConfig(
                                root_dir=root_dir,
                                base_model_path=base_model_path,
                                pretrained=self.params.PRETRAINED,
                                params_image_size=self.params.IMAGE_SIZE,
                                params_learning_rate=self.params.LEARNING_RATE,
                                params_arch_name = arch_name,
                                params_classes=self.params.CLASSES)
        
        return prepare_base_model_config