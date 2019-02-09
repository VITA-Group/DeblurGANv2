import torch.utils.data


class CustomDataLoader:
    def __init__(self, config, filename):
        self.config = config
        self.filename = filename
        self.dataset = self.create_dataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=int(config['num_workers']),
            drop_last=True)

    def name():
        return 'CustomDataLoader'

    def create_dataset(self):
        if self.config['dataset']['task'] == 'deblur':
            from data.deblur_dataset import DeblurDataset
            dataset = DeblurDataset(self.config, self.filename)
        else:
            raise ValueError("Dataset [%s] not recognized." % self.config['dataset_mode'])

        print("dataset [%s] was created" % (dataset.name()))
        return dataset

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
