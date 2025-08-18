# main.py
from pathlib import Path

import config
from pipeline_unified import run_unified_pipeline


def main():
    datasets = [("hyperkvasir", Path("./data/hyperkvasir/labeled-images/")),
                ("covidquex", Path("./data/covidquex/data/"))]

    pipeline_config = config.PipelineConfig()
    pipeline_config.file.use_cached_original = False
    pipeline_config.file.use_cached_perturbed = ""
    pipeline_config.file.current_mode = "val"
    pipeline_config.file.weighted = True
    pipeline_config.classify.analysis = False
    pipeline_config.classify.data_collection = False

    for dataset_name, source_path in datasets:
        pipeline_config.file.set_dataset(dataset_name)
        print(f"\nRunning {dataset_name} - Mode: {pipeline_config.file.current_mode}")

        run_unified_pipeline(
            config=pipeline_config,
            dataset_name=dataset_name,
            source_data_path=source_path,
            prepared_data_path=Path(f"./data/{dataset_name}_unified/"),
            force_prepare=False
        )


if __name__ == "__main__":
    main()
