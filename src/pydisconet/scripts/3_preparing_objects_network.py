import logging, torch, argparse, ast
import preprocessing
from utils import _check_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameters', help= 'Parameters', type=str, required=True)
    parser.add_argument('-s', '--save_path', help= 'Save Path', type=str, required=True)

    # Extra optional data preprocessing parameters
    data_args = parser.add_argument_group('Data split parameters', 'Set of parameters to split and preprocess the dataset')
    data_args.add_argument('-n', '--neg', help='Edge negative sampling rate', type=int, default=0)
    data_args.add_argument('-var','--val_ratio', help='Validation data ratio', type=float, default=0.2)
    data_args.add_argument('-ter','--test_ratio', help='Test data ratio', type=float, default=0.3)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = args.save_path
    year, data, component, embedding, model, control = ast.literal_eval(args.parameters)
    path = f"{save_path}/{year}/{data}/{component}/{embedding}"

    neg_sampling_ratio=args.neg
    num_val=args.val_ratio
    num_test=args.test_ratio
    preprocessing.preprocessing_for_model(path,model,num_val,num_test,neg_sampling_ratio, device)