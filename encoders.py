import gc
import re
import torch
import numpy as np
from tqdm import tqdm
from Pepnet.pepnet.encoder import Encoder
import pandas as pd
from typing import List, Tuple
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader


class ProteinEncoder:
    def __init__(self) -> None:
        """
        Initialize the ProteinEncoder with the Pepnet ecoder.
        """
        self.pepnet_encoder = Encoder()


    def get_index_embeddings(
        self,
        alleles_train: pd.Series,
        peptides_train: pd.Series,
        alleles_test: pd.Series,
        peptides_test: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Represent every amino acid with a number between 1-21 (0 is reserved for padding).

        Args:
            alleles_train (pd.Series): Training alleles.
            peptides_train (pd.Series): Training peptides.
            alleles_test (pd.Series): Testing alleles.
            peptides_test (pd.Series): Testing peptides.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Encoded training and testing alleles and peptides.
        """
        index_encoded_alleles_train = np.array(self.pepnet_encoder.encode_index_array(alleles_train.values, max_peptide_length=40))
        index_encoded_alleles_test = np.array(self.pepnet_encoder.encode_index_array(alleles_test.values, max_peptide_length=40))

        index_encoded_peptides_train = np.array(self.pepnet_encoder.encode_index_array(peptides_train.values, max_peptide_length=15))
        index_encoded_peptides_test = np.array(self.pepnet_encoder.encode_index_array(peptides_test.values, max_peptide_length=15))

        return index_encoded_alleles_train, index_encoded_peptides_train, index_encoded_alleles_test, index_encoded_peptides_test


    def get_fofe_embeddings(
        self,
        alleles_train: pd.Series,
        peptides_train: pd.Series,
        alleles_test: pd.Series,
        peptides_test: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get FOFE (Fixed-Size Ordinally Forgetting Encoding) embeddings for alleles and peptides.

        Args:
            alleles_train (pd.Series): Training alleles.
            peptides_train (pd.Series): Training peptides.
            alleles_test (pd.Series): Testing alleles.
            peptides_test (pd.Series): Testing peptides.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: FOFE encoded training and testing alleles and peptides.
        """
        fofe_encoded_alleles_train = np.array(self.pepnet_encoder.encode_FOFE(alleles_train.values, bidirectional=True))
        fofe_encoded_alleles_test = np.array(self.pepnet_encoder.encode_FOFE(alleles_test.values, bidirectional=True))

        fofe_encoded_peptides_train = np.array(self.pepnet_encoder.encode_FOFE(peptides_train.values, bidirectional=True))
        fofe_encoded_peptides_test = np.array(self.pepnet_encoder.encode_FOFE(peptides_test.values, bidirectional=True))

        return fofe_encoded_alleles_train, fofe_encoded_peptides_train, fofe_encoded_alleles_test, fofe_encoded_peptides_test


    def get_blosum_embeddings(
        self,
        alleles_train: pd.Series,
        peptides_train: pd.Series,
        alleles_test: pd.Series,
        peptides_test: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get BLOSUM (BLOcks SUbstitution Matrix) embeddings for alleles and peptides.

        Args:
            alleles_train (pd.Series): Training alleles.
            peptides_train (pd.Series): Training peptides.
            alleles_test (pd.Series): Testing alleles.
            peptides_test (pd.Series): Testing peptides.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: BLOSUM encoded training and testing alleles and peptides.
        """
        blosum_encoded_alleles_train = self.pepnet_encoder.encode_blosum(alleles_train.values, max_peptide_length=35)
        blosum_encoded_alleles_train = np.array([np.array(peptide).flatten() for peptide in blosum_encoded_alleles_train], dtype=np.float32)

        blosum_encoded_alleles_test = self.pepnet_encoder.encode_blosum(alleles_test.values, max_peptide_length=35)
        blosum_encoded_alleles_test = np.array([np.array(peptide).flatten() for peptide in blosum_encoded_alleles_test], dtype=np.float32)

        blosum_encoded_peptides_train = self.pepnet_encoder.encode_blosum(peptides_train.values, max_peptide_length=15)
        blosum_encoded_peptides_train = np.array([np.array(peptide).flatten() for peptide in blosum_encoded_peptides_train], dtype=np.float32)

        blosum_encoded_peptides_test = self.pepnet_encoder.encode_blosum(peptides_test.values, max_peptide_length=15)
        blosum_encoded_peptides_test = np.array([np.array(peptide).flatten() for peptide in blosum_encoded_peptides_test], dtype=np.float32)

        return blosum_encoded_alleles_train, blosum_encoded_peptides_train, blosum_encoded_alleles_test, blosum_encoded_peptides_test


    def get_embeddings(
        self,
        method: str,
        alleles_train: pd.Series,
        peptides_train: pd.Series,
        alleles_test: pd.Series,
        peptides_test: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get embeddings for alleles and peptides based on the specified method.

        Args:
            method (str): The encoding method ('index', 'fofe', 'blosum').
            alleles_train (pd.Series): Training alleles.
            peptides_train (pd.Series): Training peptides.
            alleles_test (pd.Series): Testing alleles.
            peptides_test (pd.Series): Testing peptides.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Encoded training and testing alleles and peptides.

        Raises:
            ValueError: If the specified method is not recognized.
        """
        prefix_to_function = {
            'index': self.get_index_embeddings,
            'fofe': self.get_fofe_embeddings,
            'blosum': self.get_blosum_embeddings
        }
        function = prefix_to_function.get(method)
        if function:
            return function(alleles_train, peptides_train, alleles_test, peptides_test)
        else:
            raise ValueError(f"Function for method '{method}' not found")
        


class ProttransEncoder:
    def __init__(self) -> None:
        """
        Initialize the ProttransEncoder with the T5 model and tokenizer.
        """
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)


    def preprocess_sequences(self, sequences: List[str]) -> List[str]:
        """
        Preprocess protein sequences by map rarely occured amino acids (U,Z,O,B) to (X) adding spaces between amino acids.

        Args:
            sequences (List[str]): List of protein sequences.

        Returns:
            List[str]: List of preprocessed protein sequences.
        """
        return [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]


    def encode_sequences(self, loader: DataLoader, device: str = 'cpu', keep_dict: bool = False) -> np.ndarray:
        """
        Encode protein sequences into embeddings using the T5 model.

        Args:
            loader (List[str]): List of protein sequences.
            device (str): Device to run the model on (default is 'cpu').
            keep_dict (bool): Whether to keep a dictionary of sequences and their embeddings (default is False). To use only with allele sequences because not many unique values. 

        Returns:
            np.ndarray: Array of protein sequence embeddings.
        """
        self.model.eval()
        self.model.to(device)
        embeddings = []
        sequence_dict = {} if keep_dict else None

        with torch.no_grad():
            for seqs in tqdm(loader):
                if keep_dict and seqs[0] in sequence_dict:
                    embeddings.append(sequence_dict[seqs[0]])
                    continue
                ids = self.tokenizer(seqs, add_special_tokens=True, padding="longest")
                input_ids = torch.tensor(ids['input_ids']).to(device)
                attention_mask = torch.tensor(ids['attention_mask']).to(device)
                embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)

                for i in range(len(seqs)):
                    embd = embedding_repr.last_hidden_state[i, :len(seqs[i])].mean(dim=0).cpu().numpy()
                    embeddings.append(embd)
                    if keep_dict:
                        sequence_dict[seqs[0]] = embd

        gc.collect()
        return np.vstack(embeddings)