import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(config, filename):
    if config['dataset']['mode'] == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % config['dataset_mode'])

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(config, filename)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, config, filename):
        BaseDataLoader.initialize(self, config)
        self.dataset = CreateDataset(config, filename)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=int(config['num_workers']),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
