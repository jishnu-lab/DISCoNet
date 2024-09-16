import logging
def combining_files(input_files, output_file):
    with open(output_file, 'a') as output_handle:
        first_file = True
        for input_file in input_files:
            with open(input_file, 'r') as input_handle:
                if not first_file:
                    next(input_handle)  # Skip the header
                for line in input_handle:
                    output_handle.write(line)
            first_file = False
    return None

def process_combine_years(save_path, years_list, data):
    combined_years_dict = {year for year in years_list if '_' in year}
    if len(combined_years_dict) > 0:
        for year in tqdm(combined_years_dict, desc='Saving combined year dataframes'):
            start_year = year.split('_')[1] 
            end_year = year.split('_')[2]
            if data =='openalex':
                input_files = [f'{save_path}/{yr}/openalex/{yr}_journal_filtered.csv' for yr in range(int(start_year), int(end_year)+1)]
                output_file = f'{save_path}/{year}/openalex/{year}_journal_filtered.csv'
                combining_files(input_files, output_file)
                logging.info(f"Data has been written to {save_path}/{year}/openalex/{year}_journal_filtered.csv")
            elif data == 'arxiv':
                input_files = [f'{save_path}/{yr}/arxiv/{yr}.csv' for yr in range(int(start_year), int(end_year)+1)]
                output_file = f'{save_path}/{year}/arxiv/{year}.csv'
                combining_files(input_files, output_file)
                logging.info(f"Data has been written to {save_path}/{year}/arxiv/{year}.csv")
    else:
        logging.error("No combined years found in the list. Please check the YEARS list and try again. Exiting...")
    return None