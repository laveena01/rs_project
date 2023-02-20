from torchvision import datasets, transforms
from new_methods.data.dior import *
from new_methods.data.NWPU_VHR import *

map_size = {'448': 512, '224': 256, '672': 768}
def get_data(name = 'DIOR', output_size = '448', split='train'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(int(output_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(map_size[output_size]),
            transforms.CenterCrop(int(output_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if name == 'DIOR':
        datasets = DIORDataset(image_set=split, transform=data_transforms[split])
    elif name == 'VHR10':
        datasets = VHR_10(image_set=split, transform=data_transforms[split])
    else:
        from new_methods.data.airplane_data import train_dataset, classes
        datasets = train_dataset
        categories = classes
    if name == 'airplane':
        return datasets, categories
    return datasets, None

def get_dataloader(name = 'DIOR', output_size = '448', split='train',shuffle=True ,batch_size=4):
    datasets, categories = get_data(name, output_size, split)
    dataset_sizes = len(datasets)
    dataloaders = torch.utils.data.DataLoader(datasets, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=2)
    if name == 'DIOR' or name == 'VHR10':
        return dataloaders, dataset_sizes, datasets.categories
    return dataloaders, dataset_sizes,categories