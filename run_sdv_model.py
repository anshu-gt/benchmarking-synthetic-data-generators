import pandas
import os
import time
import logging
import pickle
import tracemalloc
import pandas as pd
from datetime import datetime

# tabular
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import GaussianCopulaSynthesizer

# sequential
# from sdv.sequential import PARSynthesizer

# autodetection of metadata
from sdv.metadata import SingleTableMetadata

LOGGER = logging.getLogger(__name__)

# configs
EVALUATE_DATA_MODALITY = "tabular"  # valid values -- sequential, free_text

VALID_DATA_MODALITIES = ["tabular", "sequential", "free_text"]

assert EVALUATE_DATA_MODALITY in VALID_DATA_MODALITIES

# TODO: incomplete listing
EVALUATE_SYNTHESIZERS = {
    "tabular": ["ctgan", "tvae"],  # "gaussian_copula"],
    "sequential": ["par", "dgan"],
    "free_text": ["gpt", "lstm"]
}

SYNTHESIZER_MAPPING = {
    "ctgan": CTGANSynthesizer,
    "tvae": TVAESynthesizer,
    "gaussian_copula": GaussianCopulaSynthesizer
}

# TODO: incomplete listing
# datasets for each modality
EVALUATE_DATASETS = {
    "tabular": ["adult"],
    "sequential": [],
    "free_text": []
}


# static values
N_BYTES_IN_MB = 1000 * 1000

# for s3
S3_PREFIX = 's3://'
# BASE_PATH = "s3://{S3_DATA_BUCKET}/"

BASE_PATH = "data"

synthesizers = EVALUATE_SYNTHESIZERS[EVALUATE_DATA_MODALITY]
datasets = EVALUATE_DATASETS[EVALUATE_DATA_MODALITY]


use_gpu = False


def detect_metadata(real_data_df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data_df)
    # pprint(metadata.to_dict())
#     python_dict = metadata.to_dict()
    return metadata


for synthesizer_name in synthesizers:
    synthesizer_class = SYNTHESIZER_MAPPING[synthesizer_name]

    # ---------------------
    execution_scores = {
        'Synthesizer': [],
        'Dataset': [],
        # 'Dataset_Size_MB': [],
        'Train_Time': [],
        'Peak_Memory_MB': [],
        'Synthesizer_Size_MB': [],
        'Sample_Time': [],
        # 'Evaluate_Time': [],
    }

    for dataset_name in datasets:

        dataset_path = f"{BASE_PATH}/{EVALUATE_DATA_MODALITY}/{dataset_name}/{dataset_name}.csv"

        if S3_PREFIX in dataset_path:
            # data = load_dataset_from_s3(data_path)
            ...

        data = pd.read_csv(dataset_path)

        real_data = data.copy()
        num_samples = len(real_data)

        metadata = detect_metadata(real_data)

        tracemalloc.start()

        # metadata: ingle table metadata representing the data that this synthesizer will be used for.
        # cuda (bool or str):
        #             If ``True``, use CUDA. If a ``str``, use the indicated device.
        #             If ``False``, do not use cuda at all.
        # enforce_min_max_values (bool):
        #             Specify whether or not to clip the data returned by ``reverse_transform`` of
        #             the numerical transformer, ``FloatFormatter``, to the min and max values seen
        #             during ``fit``. Defaults to ``True``.
        # enforce_rounding (bool):
        #             Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
        #             by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.

        synthesizer = synthesizer_class(
            metadata, enforce_rounding=False,  epochs=2, cuda=use_gpu)

        # synthesizer = CTGANSynthesizer(
        #         metadata, # required
        #         enforce_rounding=False,
        #         epochs=500,
        #         verbose=True
        #     )

        begin_time = time.time()  # datetime.utcnow()
        # ---------------------
        # Train
        # ---------------------
        synthesizer.fit(real_data)
        train_time = time.time()

        synthesizer_size = len(pickle.dumps(synthesizer)) / N_BYTES_IN_MB

        #
        output_path = f"output/{EVALUATE_DATA_MODALITY}/{synthesizer_name}/{dataset_name}/"

        synthesizer.save(
            filepath=f'{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pkl'
        )

        # ---------------------
        # Sample
        # ---------------------
        synthetic_data = synthesizer.sample(num_rows=num_samples)
        sampling_time = time.time()

        peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB
        tracemalloc.stop()
        tracemalloc.clear_traces()

        execution_scores["Synthesizer"].append(synthesizer_name)
        execution_scores["Dataset"].append(dataset_name)
        execution_scores["Train_Time"].append(train_time - begin_time)
        execution_scores["Peak_Memory_MB"].append(peak_memory)
        execution_scores["Synthesizer_Size_MB"].append(synthesizer_size)
        execution_scores["Sample_Time"].append(sampling_time - train_time)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        synthetic_data.to_csv(
            output_path + f"{dataset_name}_{synthesizer_name}_synthetic_data.csv")

    execution_scores_df = pd.DataFrame(execution_scores)
    execution_scores_df.to_csv(output_path + "eval.csv")
#


# if __name__ == "__main__":
#     # Redirect print output to a file
#     with open("output.txt", "w") as f:
#         # Replace "output.txt" with the name of the file you want to save the print output to
#         # The "w" mode will create or overwrite the file if it already exists.
#         # If you want to append to an existing file, use "a" mode instead.

#         # Redirect the standard output (stdout) to the file
#         import sys
#         original_stdout = sys.stdout
#         sys.stdout = f

#         # Call the main function (this will execute the code with print statements)
#         main()

#         # Restore the standard output (stdout) to the original value
#         sys.stdout = original_stdout
