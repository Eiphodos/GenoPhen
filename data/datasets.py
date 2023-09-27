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


class GenoPhenoFTDataset_legacy(Dataset):
    def __init__(self, cfg, dataframe, tokenizer_geno, tokenizer_pheno):
        self.ab_index_list = cfg['antibiotics']['index_list']
        self.tokenizer_geno = tokenizer_geno
        # self.labels = labels
        self.data = dataframe
        self.tokenizer_pheno = tokenizer_pheno
        self.test_count = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data.iloc[idx]
        geno_x = sample_data['AMR_genotypes_core']
        ab_x = sample_data['AST_phenotypes_x']
        ab_y = sample_data['AST_phenotypes_y']
        gene_words = geno_x.split(',')
        ab_x = ab_x.split(',')
        ab_y = ab_y.split(',')
        random.shuffle(gene_words)

        meta_data = []
        if 'geo_loc_name' in self.data.columns:
            meta_data.append(sample_data['geo_loc_name'])
        if 'gender' in self.data.columns:
            meta_data.append(sample_data['gender'])
        else:
            meta_data.append(random.choice(['M', 'F']))

        x = ['<cls>'] + meta_data + ['unk'] + ab_x + (
                    (14 - len(ab_x)) * ['<pad>'])  # the metadata and antibiotics known

        x = self.tokenizer_pheno(x)
        total_len_x = len(ab_x) + 2 + len(meta_data)  # +5 is due to the 4 metadata entries and the cls entry

        x_pos_antibiotic = self.get_ab_pos(ab_x, "x")[0:14]  # the position of the antibiotics in abbreviation list
        y_pos_antibiotic = self.get_ab_pos(ab_y, "y")[0:8]

        x_resp = self.get_ab_resp(ab_x, "x")[0:14]  # numeric interpretation of each r and s
        y_resp = self.get_ab_resp(ab_y, "y")[0:8]

        len_x = len(ab_x)
        len_y = len(ab_y)

        if len_y > 8:  # some exceptions were found with more than 8 unknown
            print("Found example with more than 8 unknown! {}".format(ab_y))
            len_y = 8

        amr = self.tokenizer_geno.encode(gene_words)

        return {'input_ids': amr,  # genotype data
                "ab": x,  # phenotype data
                'x pos ab': x_pos_antibiotic,
                'y pos ab': y_pos_antibiotic,
                'x resp': x_resp,
                'y resp': y_resp,
                'len x': len_x,
                'len y': len_y,
                "total len x": total_len_x}

    def get_ab_pos(self, ab_list, letter):  # letter indicates if ab is known (x) or unknown (y)
        pos_list = []
        for i in range(len(ab_list)):
            abbrev = ab_list[i][0:3]  # the abbreviation of each antibiotic, one at a time
            for j in range(len(self.ab_index_list)):
                if abbrev == self.ab_index_list[j]['abbrev']:
                    pos_list.append(j)  # the position of this antibiotic in our abbrevation list
        if letter == "y":
            pos_list += [-1] * (8 - len(pos_list))  # fill upp so that we have 8 elements in our unknown vector
        if letter == "x":
            pos_list += [-1] * (14 - len(pos_list))  # fill upp so that we have 14 elements in our unknown vector
        return pos_list


    def get_ab_resp(self, ab_list, letter):
        resp_list = []
        for i in range(len(ab_list)):
            resp = ab_list[i][-1]  # the letter indicating S or R
            if resp == "S":
                resp_list.append(0)
            else:
                resp_list.append(1)
        if letter == "y":
            resp_list += [-1] * (8 - len(resp_list))
        if letter == "x":
            resp_list += [-1] * (14 - len(resp_list))
        return resp_list