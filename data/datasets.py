import random
import numpy as np
from torch.utils.data import Dataset
import torch


class GenoPTDataset(Dataset):
    def __init__(self, data, tokenizer, hierarchy_variant=None, max_n_hier=9):
        self.tokenizer = tokenizer
        self.data = data
        self.hierarchy_variant = hierarchy_variant
        self.max_n_hier = max_n_hier


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        all_words = []
        data_dict = {}
        if self.hierarchy_variant is not None:
            if self.hierarchy_variant == 'summed':
                g = self.data['AMR_genotypes_core'].iloc[idx]
                h = self.data['Hierarchy_data'].iloc[idx]

                gene_words = g.split(',')
                h = h.split(',')
                z = list(zip(gene_words, h))
                random.shuffle(z)
                gene_words, h = zip(*z)

                h = [self.tokenizer.encode(a.split(';'), add_special_tokens=False) for a in h]
                h = [[self.tokenizer.cls_token_id]*self.max_n_hier] + h + [[self.tokenizer.eos_token_id]*self.max_n_hier]
                data_dict['gene_ids'] = h
            elif self.hierarchy_variant == 'separate':
                w = self.data['Hierarchy_data'].iloc[idx]
                w = w.split(',')
                n_genes = len(w)
                h = [h.split(';') + ['<gpsep>'] for h in w]
                gi = [[i+1] * len(h[i]) for i in range(n_genes)]
                flat_h = [a for b in h for a in b]
                flat_gi = [a for b in gi for a in b]
                gene_words = flat_h
                data_dict['gene_ids'] = [1] + flat_gi + [n_genes]
        else:
            w = self.data['AMR_genotypes_core'].iloc[idx]
            w = w.split(',')
            gene_words = w
            random.shuffle(gene_words)
        tokenized_words = self.tokenizer.encode(gene_words, add_special_tokens=False)
        tokenized_words = [self.tokenizer.cls_token_id] + tokenized_words
        data_dict['input_ids'] = tokenized_words
        return data_dict


class GenoPTAllGenesDataset(Dataset):
    def __init__(self, data, tokenizer, unique_genes, gene_probs):
        self.tokenizer = tokenizer
        self.data = data
        self.unique_genes = unique_genes
        self.gene_probs = np.array(gene_probs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = {}

        g = self.data['AMR_genotypes_core'].iloc[idx]
        gi = self.data['existing_genes'].iloc[idx]
        g = g.split(',')
        ng = len(g)
        gene_exist = [1]*ng


        probs = self.gene_probs.copy()
        probs[gi] = 0.0
        probs = probs * (1 / probs.sum()) # Renormalize so that sum is equal to 1 again.
        genes_not_existing = list(np.random.choice(self.unique_genes, size=ng, replace=False, p=probs))
        gene_not_exist = [0]*ng

        g = g + genes_not_existing
        gene_xist_emb = gene_exist + gene_not_exist

        z = list(zip(g, gene_xist_emb))
        random.shuffle(z)
        g, gene_xist_emb = zip(*z)

        tokenized_words = [self.tokenizer.cls_token_id] + self.tokenizer.encode(g, add_special_tokens=False)
        gene_xist_emb = gene_xist_emb
        data_dict['input_ids'] = tokenized_words
        data_dict['gene_ids'] = gene_xist_emb
        return data_dict


class GenoPTRandomGenesDataset(Dataset):
    def __init__(self, data, tokenizer, unique_genes, gene_probs, n_genes):
        self.tokenizer = tokenizer
        self.data = data
        self.unique_genes = unique_genes
        self.gene_probs = np.array(gene_probs)
        self.n_genes = n_genes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = {}

        g = self.data['AMR_genotypes_core'].iloc[idx]
        g = g.split(',')

        random_genes = list(np.random.choice(self.unique_genes, size=self.n_genes, replace=False, p=self.gene_probs))
        gene_exist_emb = [1 if a in g else 0 for a in random_genes]

        z = list(zip(random_genes, gene_exist_emb))
        random.shuffle(z)
        random_genes, gene_exist_emb = zip(*z)
        random_genes = list(random_genes)
        gene_exist_emb = list(gene_exist_emb)
        
        existing_genes_to_guess = [b for b in g if b not in random_genes]
        if len(existing_genes_to_guess) > 0:
            n_gtg = len(existing_genes_to_guess)
            inc_genes = list(random_genes) + existing_genes_to_guess
            updated_probs = np.array([p if p not in inc_genes else 0.0 for p in self.gene_probs])
            updated_probs = updated_probs * (1 / updated_probs.sum())
            not_existing_genes_to_guess = list(np.random.choice(self.unique_genes, size=n_gtg,
                                                                replace=False, p=updated_probs))
            genes_to_guess_emb = [1]*n_gtg + [0]*n_gtg
            genes_to_guess = existing_genes_to_guess + not_existing_genes_to_guess

            z = list(zip(genes_to_guess, genes_to_guess_emb))
            random.shuffle(z)
            genes_to_guess, genes_to_guess_emb = zip(*z)
            genes_to_guess = list(genes_to_guess)
            genes_to_guess_emb = list(genes_to_guess_emb)

            all_genes = random_genes + genes_to_guess
            gene_ids = gene_exist_emb + genes_to_guess_emb
        else:
            all_genes = random_genes
            gene_ids = gene_exist_emb

        tokenized_words = [self.tokenizer.cls_token_id] + self.tokenizer.encode(all_genes, add_special_tokens=False)
        data_dict['input_ids'] = tokenized_words
        data_dict['gene_ids'] = gene_ids
        return data_dict

class GenoPhenoFTDataset_legacy(Dataset):
    def __init__(self, cfg, dataframe, tokenizer_geno, tokenizer_pheno, hierarchy_variant=None, max_n_hier=9):
        self.ab_index_list = cfg['antibiotics']['index_list']
        self.tokenizer_geno = tokenizer_geno
        # self.labels = labels
        self.data = dataframe
        self.tokenizer_pheno = tokenizer_pheno
        self.test_count = 0
        self.max_unknown_ab = 12
        self.max_total_ab = 15
        self.hierarchy_variant = hierarchy_variant
        self.max_n_hier = max_n_hier

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data.iloc[idx]
        data_dict = {}

        ### Get Genotype data ###
        if self.hierarchy_variant is not None:
            if self.hierarchy_variant == 'summed':
                g = sample_data['AMR_genotypes_core']
                h = sample_data['Hierarchy_data']

                gene_words = g.split(',')
                h = h.split(',')
                z = list(zip(gene_words, h))
                random.shuffle(z)
                gene_words, h = zip(*z)

                h = [self.tokenizer_geno.encode(a.split(';'), add_special_tokens=False) for a in h]
                h = [[self.tokenizer_geno.bos_token_id]*self.max_n_hier] + h + [[self.tokenizer_geno.eos_token_id]*self.max_n_hier]
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
            resp_list += [-1] * (self.max_total_ab - len(resp_list))
        return resp_list


class GenoPhenoFTDatasetGeneExist(Dataset):
    def __init__(self, cfg, dataframe, tokenizer_geno, tokenizer_pheno, unique_genes, gene_probs, n_genes, gene_mode, gene_only_include=None):
        self.ab_index_list = cfg['antibiotics']['index_list']
        self.tokenizer_geno = tokenizer_geno
        # self.labels = labels
        self.data = dataframe
        self.tokenizer_pheno = tokenizer_pheno
        self.test_count = 0
        self.max_unknown_ab = 12
        self.max_total_ab = 15
        self.unique_genes = unique_genes
        self.gene_probs = np.array(gene_probs)
        self.n_genes = n_genes
        self.gene_mode = gene_mode
        self.gene_only_include = gene_only_include

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data.iloc[idx]
        data_dict = {}

        ### Get Genotype data ###
        geno_x = self.data['AMR_genotypes_core'].iloc[idx]
        geno_x = geno_x.split(',')

        if self.gene_mode == 'allrandom':
            random_genes = list(np.random.choice(self.unique_genes, size=self.n_genes, replace=False, p=self.gene_probs))
            gene_exist_emb = [1 if a in geno_x else 0 for a in random_genes]

            z = list(zip(random_genes, gene_exist_emb))
            random.shuffle(z)
            random_genes, gene_exist_emb = zip(*z)
            all_genes = list(random_genes)
            gene_exist_emb = list(gene_exist_emb)
        elif self.gene_mode == 'known':
            all_genes = geno_x
            random.shuffle(all_genes)
            gene_exist_emb = [1 for g in all_genes]
        elif self.gene_mode == 'include':
            all_genes = self.gene_only_include
            gene_exist_emb = [1 if a in geno_x else 0 for a in all_genes]


        tokenized_genes = [self.tokenizer_geno.cls_token_id] + self.tokenizer_geno.encode(all_genes, add_special_tokens=False)
        data_dict['input_ids'] = tokenized_genes
        data_dict['gene_ids'] = gene_exist_emb

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
            resp_list += [-1] * (self.max_total_ab - len(resp_list))
        return resp_list