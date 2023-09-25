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
    def __init__(self, cfg, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data.iloc[idx]
        all_words = []
        geno_x = sample_data['AMR_genotypes_core']
        ab_x = sample_data['AST_phenotypes_x']
        ab_y = sample_data['AST_phenotypes_y']
        if 'species_name' in self.data.columns:
            species_data = sample_data['species_name']
        geno_x = geno_x.split(',')
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


class GenoPhenoFTDataset(Dataset):
    def __init__(self, data_amr, data_ab, tokenizer, vocab, nr_of_known, no_amr=False):
        self.tokenizer = tokenizer
        # self.labels = labels
        self.data_amr, self.x_ab, self.y_ab = combinatorial_data_generator(data_amr, data_ab, nr_of_known)
        self.vocab = vocab
        self.nr_of_known = nr_of_known
        self.nr_of_known_org = nr_of_known
        self.test_count = 0
        self.no_amr = no_amr  # set this to true in order to remove genotype data

    def __len__(self):
        return len(self.data_amr)

    def __getitem__(self, idx):
        gene_sentence = self.data_amr[idx]
        gene_words = gene_sentence.split(',')
        random.shuffle(gene_words)

        ab_sentence = self.x_ab[idx]
        meta_data = ab_sentence.split(',')[:3]
        x_ab = ab_sentence.split(',')[3:]

        y_ab = self.y_ab[idx].split(',')  # the antibiotics the model will predict

        x = ['<cls>'] + meta_data + ['unk'] + x_ab + (
                    (14 - len(x_ab)) * ['<pad>'])  # the metadata and antibiotics known

        x = vocab(x)[0:19]
        total_len_x = len(x_ab) + 5  # +5 is due to the 4 metadata entries and the cls entry

        x_pos_antibiotic = get_ab_pos(x_ab, "x")[0:14]  # the position of the antibiotics in abbreviation list
        y_pos_antibiotic = get_ab_pos(y_ab, "y")[0:8]

        x_resp = get_ab_resp(x_ab, "x")[0:14]  # numeric interpretation of each r and s
        y_resp = get_ab_resp(y_ab, "y")[0:8]

        len_x = len(x_ab)
        len_y = len(y_ab)

        if len_y > 8:  # some exceptions were found with more than 8 unknown
            len_y = 8

        if self.no_amr:  # token 4 means "unknown"
            amr = [0, 4, 2]
        else:
            amr = self.tokenizer.encode(gene_words)

        self.nr_of_known = self.nr_of_known_org
        return {'input_ids': amr,  # genotype data
                "ab": x,  # phenotype data
                'x pos ab': x_pos_antibiotic,
                'y pos ab': y_pos_antibiotic,
                'x resp': x_resp,
                'y resp': y_resp,
                'len x': len_x,
                'len y': len_y,
                "total len x": total_len_x}