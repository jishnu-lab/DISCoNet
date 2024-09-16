import logging, database_parser, ast, argparse, pandas as pd
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters', help= 'Parameters', type=str, required=True)
    parser.add_argument('-r', '--read_path', help= 'Read Path', type=str, required=True)
    parser.add_argument('-s', '--save_path', help= 'Save Path', type=str, required=True)
    args = parser.parse_args()

    read_path = args.read_path
    save_path = args.save_path
    YEARS, dataset = ast.literal_eval(args.parameters)

    if dataset == 'arxiv':
        database_parser.download_arxiv(save_path, force_download=False)
        papers_parsed_dict = database_parser.process_arxiv(save_path, YEARS)
    elif dataset == 'openalex':
        urls_parsed_dict = database_parser.download_openalex(save_path, force_download=False)
        pd.DataFrame.from_dict(urls_parsed_dict, orient='index', columns=['papers']).to_csv(f'{save_path}/openalex_url_paper_counts.csv')
        papers_parsed_dict = database_parser.process_openalex(read_path, save_path, YEARS)
       
    else:
        logging.error("Only openalex and arxiv datasets are supported for now. Exiting...")
        exit()

    pd.DataFrame.from_dict(papers_parsed_dict, orient='index', columns=['papers']).to_csv(f'{save_path}/{dataset}_paper_counts.csv')
    database_parser.process_combine_years(save_path, YEARS, dataset)
    logging.info(f"{dataset} processing step completed. Exiting the script...")