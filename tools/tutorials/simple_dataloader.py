from occnet.builder import build_dataloader

version = 'v1.0-mini'

dataset_params = dict(
    version = version,
    ignore_label = 0,
    fill_label = 17,
    fixed_volume_space = True,
    label_mapping = "/github/The-Eyes-Have-It/configs/_base_/label_mapping/nuscenes-noIgnore.yaml",
    max_volume_space = [51.2, 51.2, 3],
    min_volume_space = [-51.2, -51.2, -5],
)

train_data_loader = dict(
    data_path = "/github/The-Eyes-Have-It/data/nuscenes",
    imageset = "/github/The-Eyes-Have-It/data/nuscenes_infos_train.pkl",
    batch_size = 1,
    shuffle = True,
    num_workers = 0,    # debug use
)

val_data_loader = dict(
    data_path = "/github/The-Eyes-Have-It/data/nuscenes",
    imageset = "/github/The-Eyes-Have-It/data/nuscenes_infos_val.pkl",
    batch_size = 1,
    shuffle = False,
    num_workers = 0,
)

dataloader = build_dataloader(dataset_params,
                              train_data_loader,
                              val_data_loader,
                              version=version)

train, test = dataloader
train_iter = iter(train)
sample = next(train_iter)