import random
import numpy as np
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


class GenoPTDataset(Dataset):
    def __init__(self, data, tokenizer, hierarchy_variant=None):
        self.tokenizer = tokenizer
        self.data = data
        self.hierarchy_variant = hierarchy_variant

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        all_words = []
        data_dict = {}
        if self.hierarchy_variant is not None:
            if self.hierarchy_variant == 'summed':
                g = self.data['AMR_genotypes_core'].iloc[idx]
                h = self.data['Hierarchy_data'].iloc[idx]
                all_words = g.split(',')
                h = h.split(',')
                h = [self.tokenizer.encode(a.split(';'), add_special_tokens=False) for a in h]
                h = [[self.tokenizer.bos_token_id]] + h + [[self.tokenizer.eos_token_id]]
                data_dict['gene_ids'] = h
            elif self.hierarchy_variant == 'separate':
                w = self.data['Hierarchy_data'].iloc[idx]
                w = w.split(',')
                n_genes = len(w)
                h = [h.split(';') + ['<gpsep>'] for h in w]
                gi = [[i+1] * len(h[i]) for i in range(n_genes)]
                flat_h = [a for b in h for a in b]
                flat_gi = [a for b in gi for a in b]
                all_words = flat_h
                data_dict['gene_ids'] = [1] + flat_gi + [n_genes]
        else:
            w = self.data['AMR_genotypes_core'].iloc[idx]
            w = w.split(',')
            all_words = w
            random.shuffle(all_words)
        tokenized_words = self.tokenizer.encode(all_words)
        data_dict['input_ids'] = tokenized_words
        return data_dict


class GenoPhenoFTDataset_legacy(Dataset):
    def __init__(self, cfg, dataframe, tokenizer_geno, tokenizer_pheno, hierarchy_variant=None):
        self.ab_index_list = cfg['antibiotics']['index_list']
        self.tokenizer_geno = tokenizer_geno
        # self.labels = labels
        self.data = dataframe
        self.tokenizer_pheno = tokenizer_pheno
        self.test_count = 0
        self.max_unknown_ab = 9
        self.max_total_ab = 14
        self.hierarchy_variant = hierarchy_variant

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data.iloc[idx]
        data_dict = {}

        ### Get Genotype data ###
        if self.hierarchy_variant is not None:
            if self.hierarchy_variant == 'summed':
                g = sample_data['AMR_genotypes_core']
                gene_words = g.split(',')
                h = sample_data['Hierarchy_data']
                h = h.split(',')
                random.shuffle(h)
                h = [self.tokenizer_geno.encode(a.split(';'), add_special_tokens=False) for a in h]
                h = [[self.tokenizer_geno.bos_token_id]] + h + [[self.tokenizer_geno.eos_token_id]]
                data_dict['gene_ids'] = h
            elif self.hierarchy_variant == 'separate':
                geno_x = sample_data['Hierarchy_data']
                geno_x = geno_x.split(',')
                n_genes = len(geno_x)
                h = [h.split(';') + ['<gpsep>'] for h in geno_x]
                gi = [[i+1] * len(h[i]) for i in range(n_genes)]
                flat_h = [a for b in h for a in b]
                flat_gi = [a for b in gi for a in b]
                gene_words = flat_h
                data_dict['gene_ids'] = [1] + flat_gi + [n_genes]
        else:
            geno_x = sample_data['AMR_genotypes_core']
            gene_words = geno_x.split(',')
            random.shuffle(gene_words)
            data_dict['gene_ids'] = np.nan
        amr = self.tokenizer_geno.encode(gene_words)
        data_dict['input_ids'] = amr

        ### Get Phenotype data ###
        ab_x = sample_data['AST_phenotypes_x']
        ab_y = sample_data['AST_phenotypes_y']
        meta_data = []
        if 'geo_loc_name' in self.data.columns:
            meta_data.append(sample_data['geo_loc_name'])
        else:
            meta_data.append('SE')
        if "age" in self.data.columns:
            meta_data.append(sample_data['age'])
        else:
            meta_data.append('67')
        if 'gender' in self.data.columns:
            meta_data.append(sample_data['gender'])
        else:
            meta_data.append(random.choice(['M', 'F']))
        if "target_creation_date" in self.data.columns:
            meta_data.append(sample_data['target_creation_date'])

        x = ['<cls>'] + meta_data + ['<sep>'] + ab_x + (
                    (self.max_total_ab - len(ab_x)) * ['<pad>'])  # the metadata and antibiotics known
        x = self.tokenizer_pheno(x)
        data_dict['ab'] = x

        ### Get AB positions ###
        total_len_x = len(ab_x) + 2 + len(meta_data)  # +5 is due to the 4 metadata entries and the cls entry
        x_pos_antibiotic = self.get_ab_pos(ab_x, "x")[0:self.max_total_ab]  # the position of the antibiotics in abbreviation list
        y_pos_antibiotic = self.get_ab_pos(ab_y, "y")[0:self.max_unknown_ab]
        data_dict['x pos ab'] = x_pos_antibiotic
        data_dict['y pos ab'] = y_pos_antibiotic
        data_dict['total len x'] = total_len_x

        ### Get AB representations ###
        x_resp = self.get_ab_resp(ab_x, "x")[0:self.max_total_ab]  # numeric interpretation of each r and s
        y_resp = self.get_ab_resp(ab_y, "y")[0:self.max_unknown_ab]
        data_dict['x resp'] = x_resp
        data_dict['y resp'] = y_resp

        ### Get AB lengths ###
        len_x = len(ab_x)
        len_y = len(ab_y)
        if len_y > self.max_unknown_ab:  # some exceptions were found with more than 8 unknown
            print("Found example with more than {} unknown! {}".format(self.max_unknown_ab, ab_y))
            len_y = self.max_unknown_ab
        data_dict['len x'] = len_x
        data_dict['len y'] = len_y

        return data_dict

    def get_ab_pos(self, ab_list, letter):  # letter indicates if ab is known (x) or unknown (y)
        pos_list = []
        for i in range(len(ab_list)):
            abbrev = ab_list[i][0:3]  # the abbreviation of each antibiotic, one at a time
            for j in range(len(self.ab_index_list)):
                if abbrev == self.ab_index_list[j]['abbrev']:
                    pos_list.append(j)  # the position of this antibiotic in our abbrevation list
        if letter == "y":
            pos_list += [-1] * (self.max_unknown_ab - len(pos_list))  # fill upp so that we have 8 elements in our unknown vector
        if letter == "x":
            pos_list += [-1] * (self.max_total_ab - len(pos_list))  # fill upp so that we have 14 elements in our unknown vector
        return pos_list


    def get_ab_resp(self, ab_list, letter):
        resp_list = []
        for i in range(len(ab_list)):
            resp = ab_list[i][-1]  # the letter indicating S or R
            if resp == "s":
                resp_list.append(0)
            else:
                resp_list.append(1)
        if letter == "y":
            resp_list += [-1] * (self.max_unknown_ab - len(resp_list))
        if letter == "x":
            resp_list += [-1] * (14 - len(resp_list))
        return resp_list