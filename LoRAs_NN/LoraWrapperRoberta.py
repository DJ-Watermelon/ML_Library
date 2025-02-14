import math
import torch 
import pathlib

from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import torch.nn as nn

# Basic Model Imports
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel

# Import Attentions to adjust
from torch.nn import MultiheadAttention
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

import lora_utils

from torch.nn import functional as F


class LoraRobertaSelfAttention(RobertaSelfAttention):
    """
    Extend RobertaSelfAttention with LoRA (Low-rank adaption) matrices 
    to improve efficiency by only updating query and value matrices
    LoRA logic applied to forward method.

    Args:
        r(int) : rank of LoRA matrices
        confid : Condifuration of Roberta Model 
    """
    
    def __init__(self, r=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.all_head_size
        
        # Initialize LoRA matrices for query and value
        # B initialized as zeros, A as random gaussian so product is zero
        self.lora_query_matrix_B = nn.Parameter(torch.zeros(d, r))
        self.lora_query_matrix_A = nn.Parameter(torch.randn(r, d))
        self.lora_value_matrix_B = nn.Parameter(torch.zeros(d, r))
        self.lora_value_matrix_A = nn.Parameter(torch.randn(r, d))
    
    # LoRA query logic - freeze the regular linear layer before initializing optimizer
    # Only training LoRA parameters
    def lora_query(self, x):
        lora_full_query_weights = torch.matmul(self.lora_query_matrix_B, self.lora_query_matrix_A)
        return self.query(x) + F.linear(x, lora_full_query_weights)
    
    
    # LoRA vlaue logic - freeze the regular linear layer before initializing optimizer
    # Only training LoRA parameters
    def lora_value(self, x):
        lora_full_query_weights = torch.matmul(self.lora_value_matrix_B, self.lora_value_matrix_A)
        return self.query(x) + F.linear(x, lora_full_query_weights)
    
    
    # Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
    # replaced query calls with lora_query and value calls with lora_value
    def forward(self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
        
        mixed_query_layer = self.lora_query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.lora_value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
 
 
# wrapper for RoBERTa with Low-Rank Adaptation for NLP tasks 
class LoraWrapperRoberta(nn.Module):
    def __init__(self, task_type, num_classes: int = None, dropout_rate=0.1, model_id="roberta-large",
                 lora_rank=8, train_biases=True, train_embedding=False, train_layer_norms=True):
        # task_type : str - Type of NLP task ('glue', 'squad_v1', 'squad_v2')
        # num_classes : int - Number of classes for classification
        # dropout_rate : float - Dropout rate of model
        # model_id : str - Pre-trained RoBERTa model ID
        # lora_rank : int - Rank for LoRA adaptation
        # train_biases : bool - whether to updates bias params during training 
        # train_embedding : bool - whether to update embedding layer weights during training
        # train_layer_norms: bool - whether to update the layer norms during training
        
        super().__init__()
        
        supported_task_types = ['glue', 'squad', 'squad_v1', 'squad_v2']
        assert isinstance(task_type, str) and task_type.lower() in supported_task_types, f"task_type has to be one of {supported_task_types}"
        
        # num_classes should be 2 for squad v1 and 3 for squad v2 (third : "no answer possible")
        if task_type == "squad_v1" or task_type == "squad":
            num_classes = 2
        elif task_type == "squad_v2":
            num_classes = 3
        
        # Initialize base model with parameters
        self.model_id = model_id
        self.tokenizer = RobertaTokenizer.from_pretrained(model_id)
        self.model = RobertaModel.from_pretrained(model_id)
        
        self.model_config = self.model.config # save model config to set layers
        
        self.base_model_param_count = loras_utils.count_parameters(self.model)
        
        self.lora_rank = lora_rank
        self.train_biases = train_biases
        self.train_embeddings = train_embedding
        self.train_layer_norms = train_layer_norms
        
        # Add layer for benchmark tasks
        d_model = self.model_config.hidden_size # get output size of base model
        self.d_model = d_model
        self.num_classes = num_classes
        self.task_type = task_type.lower()
        
        # Define additional norm, linear layer and dropout
        self.finetune_head_norm = nn.LayerNorm(d_model)
        self.finetune_head_dropout = nn.Dropout(dropout_rate)
        self.finetune_head_classifier = nn.Linear(d_model, num_classes)
        
        # Set up Lora model for training Benchmark task
        self.replace_multihead_attention()
        self.freeze_parameters_except_lora_and_bias()
        
    # Replace RobertaSelfAttention with LoraRobertaSelfAttention in the model
    def replace_multihead_attention(self, verbose=True):
        # verbose : bool - whether to print the number of replaced modeules
        self.nr_replaced_modules = 0
        self.replace_multihead_attention_recursion(self.model)
        if verbose:
            print(f"Replaced {self.nr_replaced_modules} modules of RobertaSelfAttention with LoraRobertaSelfAttention")
            
    # Recursively replaces RobertaSelfAttention with LoraRobertaSelfAttention
    def replace_multihead_attention_recursion(self, model):
        
        # traverse sub_components
        for name, module in model.named_children():
            if isinstance(module, RobertaSelfAttention):
                self.nr_replaced_modules += 1
                
                # Create a new LoraMultiheadAttention layer
                new_layer = LoraRobertaSelfAttention(r=self.lora_rank, config=self.model_config)
                
                # Get the state of the original layer
                state_dict_old = module.state_dict()
                
                # Load old state dict into new layer
                new_layer.load_state_dict(state_dict_old, stric=False)
                
                # Get new layer state
                state_dict_new = new_layer.state_dict()
                
                # Compare new and old state dict keys to make sure they are the same
                keys_old = set(state_dict_old.keys)
                keys_new = set(k for k in state_dict_new.keys() if not k.startswith("lora_"))
                assert keys_old == keys_new, f"Keys of the state dictionaries don't match (ignoring loras parameters):\n\tExpected Parameters: {keys_old}\n\tNew Parameters (w.o. LoRA): {keys_new}"
                
                # Replace original layer with new layer
                setattr(model, name, new_layer)
            else:
                # Recurse on child modules
                self.replace_multihead_attention_recursion(module)
        
    # Freeze all parameters in the model except LoRA layers, the finetune head and bias parameters (if specified)        
    def freeze_parameters_except_lora_and_bias(self):
        
        for name, param in self.model.named_parameters():
            if ("lora_" in name) or ("finetune_head_" in name) or (self.train_biases and "bias" in name) \
                or (self.train_embeddings and "embeddings" in name) or (self.train_layer_norms and "LayerNorm" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    # forward method for GLUE, SQuAD, and RACE Benchmarks
    def forward(self, x, attention_mask=None):
        if self.task_type == "glue":
            return self.forward_glue(x, attention_mask)
        elif self.task_type == "squad_v1" or self.task_type == "squad":
            return self.forward_squad(x, attention_mask)
        elif self.task_type == "squad_v2":
            return self.forward_squad_v2(x, attention_mask)
       
       
    def forward_glue(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask)
        # Take hidden state output from base model
        x = outputs.last_hidden_state
        x = x[:,0,:] # Get output from [CLS] token
        
        # Pass the base model output through dropout -> linear layer -> norm
        x = self.finetune_head_norm(x)
        x = self.finetune_head_dropout(x)
        x = self.finetune_head_classifier(x)
        
        if self.num_classes == 1: # for regression task
            x = torch.sigmoid(x) * 5 # scale to range [0,5] for stsb
            
        return x
    
    def forward_squad(self, x, attention_mask=None):
        # Run the input through the base model
        outputs = self.model(x, attention_mask)
        # Take hidden state output from base model
        x = outputs.last_hidden_state
        
        # Pass the base model output through dropout -> linear layer -> norm
        x = self.finetune_head_norm(x)
        x = self.finetune_head_dropout(x)
        x = self.finetune_head_classifier(x)
        
        # Flatten outputs
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits
    
    def forward_squad_v2(self, x, attention_mask=None):
        # Run the input through the base model
        outputs = self.model(x, attention_mask)
        # Take hidden state output from base model
        x = outputs.last_hidden_state
        
        # Pass the base model output through dropout -> linear layer -> norm
        x = self.finetune_head_norm(x)
        x = self.finetune_head_dropout(x)
        x = self.finetune_head_classifier(x)
        
        # Flatten outputs
        start_logits, end_logits, na_prob_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        na_prob_logits = na_prob_logits.squeeze(-1)
        
        # na_prob_logits should be a single number (START token)
        na_prob_logits = na_prob_logits[:, 0]
        
        # return the start and end logits
        return start_logits, end_logits, na_prob_logits
    
    # Save trainable parameters into a state dict.
    def save_lora_state_dict(self, lora_filepath: Optional[Union[str, Path]] = None) -> Optional[Dict]:
        # lora_filepath - file path where to save the sate dict
        
        # Create state dict of trainable parameters
        state_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}
        
        # additional parameters
        state_dict['model_id'] = self.model_id
        state_dict['task_type'] = self.task_type
        state_dict['lora_rank'] = self.lora_rank
        state_dict['num_classes'] = self.num_classes
        
        if lora_filepath is not None:
            # convert string to pathlib.Path
            if isinstance(lora_filepath, str):
                lora_filepath = Path(lora_filepath)
            # save to specified file
            torch.save(state_dict, lora_filepath)   
        else:
            return state_dict
    
    # load state dict into model from specified file path or state dict directly
    @staticmethod
    def load_lora_state_dict(lora_parameters: Union[str, Path, Dict] = None):
        if lora_parameters is not None:
            # Convert string to pathlib.Path if necessary
            if isinstance(lora_parameters, str):
                lora_parameters = Path(lora_parameters)
            
            # If the provided object is a Path, load the state dict from file
            if isinstance(lora_parameters, Path):
                state_dict = torch.load(lora_parameters)
            else:
                # Otherwise assume its a state dict
                state_dict = lora_parameters
        else:
            raise ValueError("No filepath or state dict provided")
        
        instance = LoraWrapperRoberta(task_type=state_dict['task_type'], num_classes = state_dict['num_classes'], model_id = state_dict['model_id'], lora_rank = state_dict['lora_rank'])
        
        # Load the state dict into the model
        instance.load_lora_state_dict(state_dict, strict=False)
        
        return instance