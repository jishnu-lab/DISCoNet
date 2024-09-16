import logging
from . import analyze, database_parser, interpreter, plotter, preprocessing, utils, test
from .analyze import run_gat, run_zeroshot, network_statistics, embed_datasets
from .database_parser import download_arxiv, download_openalex, process_arxiv, process_openalex, process_combine_years
from .interpreter import process_restart, check_list_inputs, process_inputs, run_job, shell_script_cpu, shell_script_gpu, develop_chunks, step1, step2, step3, step4
from .plotter import tokenize_title_per_year, tokenize_title, get_quadrants, get_auc_aupr, q_auc_aupr, process_year, calc_auc_aupr_across_years, _HexCodeFromFrequencyDict
from .preprocessing import generate_author_edge_df, generate_grouped_objects, preprocessing_for_embedding, preprocessing_for_model
from .utils import _display_input_help, _default_inputs, _dump_to_json_gz, _read_from_json_gz, _check_steps

__copyright__    = 'Copyright (C) 2024 Swapnil Keshari'
__license__      = 'MIT license'
__author__       = 'Swapnil Keshari**, Zarifeh Heidari, Akash Kishore, Jishnu Das'
__author_email__ = 'swk25@pitt.edu'
__url__          = 'https://github.com/swapnilkeshari/disconet'

logger = logging.getLogger(__name__)

__all__ = [ 'run_gat', 'run_zeroshot', 'network_statistics', 'embed_datasets', \
            'download_arxiv', 'download_openalex', 'process_arxiv', 'process_openalex', 'process_combine_years', \
            'process_restart', 'check_list_inputs', 'process_inputs', 'run_job', 'shell_script_cpu', 'shell_script_gpu', 'develop_chunks', 'step1', 'step2', 'step3', 'step4', \
            'tokenize_title_per_year', 'tokenize_title', 'get_quadrants', 'get_auc_aupr', 'q_auc_aupr', 'process_year', 'calc_auc_aupr_across_years', '_HexCodeFromFrequencyDict', \
            'generate_author_edge_df', 'generate_grouped_objects', 'preprocessing_for_embedding', 'preprocessing_for_model', \
            '_display_input_help', '_default_inputs', '_dump_to_json_gz', '_read_from_json_gz', '_check_steps', \
            'test' ]
