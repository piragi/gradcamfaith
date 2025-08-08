# main.py
from pathlib import Path

import config
from pipeline_unified import run_unified_pipeline


def main(pipeline_config: config.PipelineConfig):
    print(
        f"Run pipeline with:\nCurrent Mode: {pipeline_config.file.current_mode}, {'Weighted' if pipeline_config.file.weighted else 'Unweighted'}, {'Analysis activated' if pipeline_config.classify.analysis else 'No Analysis'}"
    )
    
    # Use unified pipeline with the prepared dev set
    _ = run_unified_pipeline(
        config=pipeline_config,
        dataset_name="hyperkvasir",
        source_data_path=Path("../../gradcamfaithkvasir/gradcamfaith/hyper-kvasir/"),
        prepared_data_path=Path(f"./data/hyperkvasir_unified/"),  # Already prepared data
        force_prepare=False  # Don't re-prepare since dev set is ready
    )


if __name__ == "__main__":
    pipeline_config = config.PipelineConfig()
    pipeline_config.file.use_cached_original = False
    pipeline_config.file.use_cached_perturbed = ""
    pipeline_config.file.current_mode = "val"
    pipeline_config.file.weighted = True
    pipeline_config.classify.analysis = False
    pipeline_config.classify.data_collection = False  #not pipeline_config.file.weighted

    main(pipeline_config)
