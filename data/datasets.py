import random
from torch.utils.data import Dataset


class GenoPTDataSet(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        if 'species_name' in data.columns:
            self.multi_species = True
        else:
            self.multi_species = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        all_genes = self.data.iloc[idx].AMR_genotypes_core
        genes_as_words = all_genes.split(',')
        random.shuffle(genes_as_words)
        all_words = genes_as_words
        if self.multi_species:
            all_words += [self.data.iloc[idx].species_name]
        #all_words = ','.join(all_words)
        tokenized_words = self.tokenizer.encode(all_words)
        return {'input_ids': tokenized_words}


class GenoPhenoPTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        all_words = []
        data_dict = {}
        for c in self.data.columns:
            w = self.data[c].iloc[idx]
            if c == 'AMR_genotypes_core' or c == 'AST_phenotypes':
                w = w.split(',')
                all_words += w
            elif c == 'Hierarchy_data':
                w = w.split(',')
                hierarchy_data = {}
                tokenized_h = [";".join(self.tokenizer.encode(h.split(';'))) for h in w]
                hierarchy_data = {h.split(';')[0]: h.split(';')[1:] for h in tokenized_h}
                data_dict['hierachy_ids'] = hierarchy_data
            else:
                all_words += [w]
        random.shuffle(all_words)
        tokenized_words = self.tokenizer.encode(all_words)
        data_dict['input_ids'] = tokenized_words
        return data_dict


class GenoPhenoFTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        all_words = []
        for c in self.data.columns:
            w = self.data[c].iloc[idx]
            if c == 'AMR_genotypes_core' or c == 'AST_phenotypes':
                w = w.split(',')
                all_words += w
            else:
                all_words += [w]
        random.shuffle(all_words)
        tokenized_words = self.tokenizer.encode(all_words)
        return {'input_ids': tokenized_words}