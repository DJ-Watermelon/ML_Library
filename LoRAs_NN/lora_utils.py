# Define general non model specific utils for LoRA training

import math
import torch 
from torch import nn
from torch import Tensor

import tabulate
import tokenizers


# Counts the number of trainable parameters of a PyTorch model
def count_parameters(model):
    # model : torch.nn.Module
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Formats a number with underscore as thousand separator
def format_with_underscore(n):
    return f"{n:_}"

# Displays formatted table containing number of trainable parameters
def parameter_count_table(model, output_file_path=None, output_print=True, add_dtypes=False, show_nograd_params=False):
    # model : torch.nn.Model
    # output_file_path : str or Path - where to write table to, printed to console if None
    # output_print : bool - if True printed to console
    # add_dtypes : bool - if True outputs the dtypes  of each parameter set
    # show_nograd_paras : bool optional - If True shows which parameters are frozen (no gradient)
    
    table = [["Module", "Parameters"]] 
    if add_dtypes:
        table = [["Module", "Parameters", "dtype"]]
    
    total_params = 0
    max_len = 0
    
    for name, parameter in model.named_parameters():
        if (not parameter.requires_grad) and (not show_nograd_params): continue
        
        params = parameter.numel()
        formatted_params = format_with_underscore(params)
        max_len = max(max_len, len(formatted_params))
        
        if add_dtypes:
            table.append([str(name), formatted_params, parameter.dtype])
        else:
            table.append([str(name), formatted_params])
        total_params += params
    
    table.append(tabulate.SEPARATING_LINE)
    
    formatted_total = format_with_underscore(total_params)
    max_len = max(max_len, len(formatted_total))
    if add_dtypes:
        table.append(["TOTAL", formatted_total])
    else:
        table.append(["TOTAL", formatted_total, ''])
        
    # Right align numbers in the table
    for row in table[1:]:
        if row is not tabulate.SEPARATING_LINE:
            row[1] = row[1].rjust(max_len)
            
    tabulated_table = tabulate.tabulate(table, headers="firstrow")
    if output_file_path is not None:
        with open(output_file_path, 'w') as f:
            f.write(tabulated_table)
    if output_print:
        print(tabulated_table)
        print("")

# Get leraning rate of an optimizer
def get_lr(optimizer):
    # optimizer : torch.optim.Optimizer
    for param_group in optimizer.param_groups:
        return param_group['lr']
            
    
