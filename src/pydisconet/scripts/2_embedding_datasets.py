import ast, logging, argparse, torch, numpy as np
import preprocessing, analyze
from utils import _dump_to_json_gz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters', help= 'Parameters', type=str, required=True)
    parser.add_argument('-s', '--save_path', help= 'Save Path', type=str, required=True)
    parser.add_argument('-b', '--batch_size', help= 'Batch size for the data streaming', type = int, default=256)
    args = parser.parse_args()

    save_path = args.save_path
    batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    year, dataset, component, embedding, model, control = ast.literal_eval(args.parameters)
    path=f'{save_path}/{year}/{dataset}/'

    filtered_grouped_work, edge_df, author_df = preprocessing.preprocessing_for_embedding(save_path, year, dataset)
    _dump_to_json_gz(edge_df.to_json(orient='records'), f'{path}/edge_df.json.gz')
    _dump_to_json_gz(author_df.to_json(orient='records'), f'{path}/author_df.json.gz')
        
    logging.info(f"Started embedding for {path}")
    paper_titles = list(set((filtered_grouped_work.work_name.dropna())))
    _dump_to_json_gz(paper_titles, f'{path}/paper_titles.json.gz')
    author_df.work_name = author_df.work_name.apply(lambda x: '. '.join(map(str,x)))

    dataset_object = analyze.embed_and_save(path, component, embedding, paper_titles, edge_df, author_df, batch_size, device)
    analyze.calc_save_network_statistics(path,component, dataset_object)
    logging.info(f"Finished network_statistics analysis for {path}")