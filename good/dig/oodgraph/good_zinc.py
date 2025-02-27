"""
The GOOD-ZINC dataset. Adapted from `ZINC database
<https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_.
"""
import os
import os.path as osp

import gdown
import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip


class GOODZINC(InMemoryDataset):
    r"""
    The GOOD-ZINC dataset adapted from `ZINC database
    <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'scaffold' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False):

        self.name = self.__class__.__name__
        self.mol_name = 'ZINC'
        self.domain = domain
        self.metric = 'MAE'
        self.task = 'Regression'
        self.url = 'https://drive.google.com/file/d/1CHR0I1JcNoBqrqFicAZVKU3213hbsEPZ/view?usp=sharing'

        self.generate = generate

        super().__init__(root, transform, pre_transform)
        if shift == 'covariate':
            subset_pt = 3
        elif shift == 'concept':
            subset_pt = 8
        elif shift == 'no_shift':
            subset_pt = 0
        else:
            raise ValueError(f'Unknown shift: {shift}.')
        if subset == 'train':
            subset_pt += 0
        elif subset == 'val':
            subset_pt += 1
        elif subset == 'test':
            subset_pt += 2
        elif subset == 'id_val':
            subset_pt += 3
        else:
            subset_pt += 4

        self.data, self.slices = torch.load(self.processed_paths[subset_pt])

    @property
    def raw_dir(self):
        return osp.join(self.root)

    def _download(self):
        if os.path.exists(osp.join(self.raw_dir, self.name)) or self.generate:
            return
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        self.download()

    def download(self):
        path = gdown.download(self.url, output=osp.join(self.raw_dir, self.name + '.zip'), fuzzy=True)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return ['no_shift_train.pt', 'no_shift_val.pt', 'no_shift_test.pt',
                'covariate_train.pt', 'covariate_val.pt', 'covariate_test.pt', 'covariate_id_val.pt',
                'covariate_id_test.pt',
                'concept_train.pt', 'concept_val.pt', 'concept_test.pt', 'concept_id_val.pt', 'concept_id_test.pt']

    @staticmethod
    def load(dataset_root: str, domain: str, shift: str = 'no_shift', generate: bool = False):
        r"""
        A staticmethod for dataset loading. This method instantiates dataset class, constructing train, id_val, id_test,
        ood_val (val), and ood_test (test) splits. Besides, it collects several dataset meta information for further
        utilization.

        Args:
            dataset_root (str): The dataset saving root.
            domain (str): The domain selection. Allowed: 'degree' and 'time'.
            shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
            generate (bool): The flag for regenerating dataset. True: regenerate. False: download.

        Returns:
            dataset or dataset splits.
            dataset meta info.
        """
        meta_info = Munch()
        meta_info.dataset_type = 'mol'
        meta_info.model_level = 'graph'

        train_dataset = GOODZINC(root=dataset_root,
                                 domain=domain, shift=shift, subset='train', generate=generate)
        id_val_dataset = GOODZINC(root=dataset_root,
                                  domain=domain, shift=shift, subset='id_val',
                                  generate=generate) if shift != 'no_shift' else None
        id_test_dataset = GOODZINC(root=dataset_root,
                                   domain=domain, shift=shift, subset='id_test',
                                   generate=generate) if shift != 'no_shift' else None
        val_dataset = GOODZINC(root=dataset_root,
                               domain=domain, shift=shift, subset='val', generate=generate)
        test_dataset = GOODZINC(root=dataset_root,
                                domain=domain, shift=shift, subset='test', generate=generate)
        train_dataset.data.y = train_dataset.data.y.reshape(-1, 1)
        if id_val_dataset:
            id_val_dataset.data.y = id_val_dataset.data.y.reshape(-1, 1)
            id_test_dataset.data.y = id_test_dataset.data.y.reshape(-1, 1)
        val_dataset.data.y = val_dataset.data.y.reshape(-1, 1)
        test_dataset.data.y = test_dataset.data.y.reshape(-1, 1)

        meta_info.dim_node = train_dataset.num_node_features
        meta_info.dim_edge = train_dataset.num_edge_features

        meta_info.num_envs = torch.unique(train_dataset.data.env_id).shape[0]

        # Define networks' output shape.
        if train_dataset.task == 'Binary classification':
            meta_info.num_classes = train_dataset.data.y.shape[1]
        elif train_dataset.task == 'Regression':
            meta_info.num_classes = 1
        elif train_dataset.task == 'Multi-label classification':
            meta_info.num_classes = torch.unique(train_dataset.data.y).shape[0]

        # --- clear buffer dataset._data_list ---
        train_dataset._data_list = None
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None
        val_dataset._data_list = None
        test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'val': val_dataset, 'test': test_dataset, 'task': train_dataset.task,
                'metric': train_dataset.metric}, meta_info
