import random
from torch.utils.data import Dataset


class GenoDataSet(Dataset):
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