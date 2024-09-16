import torch, logging, argparse, ast
import analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters', help= 'Parameters', type=str, required=True)
    parser.add_argument('-s', '--save_path', help= 'Save Path', type=str, required=True)
    
    # Extra optional model specific parameters
    model_args = parser.add_argument_group('Model specific parameters', 'Set of parameters to run zeroshot')
    model_args.add_argument('-b', '--batch_size', help='Batch size for the data loader', type=int, default=2048)
    model_args.add_argument('-n', '--neg', help='Edge negative sampling ratio', type=int, default=1)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = args.save_path
    year, data, component, embedding, model, control = ast.literal_eval(args.parameters)
    batch_size = args.batch_size
    neg_ratio = args.neg
    path = f"{save_path}/{year}/{data}/{component}/{embedding}/{model}"

    if 'zeroshot' not in model:
            logging.error("This scripts only supports 'zeroshot'. 'zeroshot' not found in models. Hence skipping.")
            exit()
    all_data, _, _, test_data = analyze._read_data(path,device)
    if control is not None:
        path = f"{path}/{control}"
    analyze.run_zeroshot(path, all_data,test_data,batch_size,neg_ratio,device,control=control)