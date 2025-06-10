# main.py
from pathlib import Path

import config
import pipeline as pipe


def main(pipeline_config: config.PipelineConfig):
    print(
        f"Run pipeline with:\nCurrent Mode: {pipeline_config.file.current_mode}, {'Weighted' if pipeline_config.file.weighted else 'Unweighted'}, {'Analysis activated' if pipeline_config.classify.analysis else 'No Analysis'}"
    )
    _ = pipe.run_pipeline(
        pipeline_config,
        source_dir_for_preprocessing=Path(f"./hyper-kvasir/preprocessed/{pipeline_config.file.current_mode}")
    )


if __name__ == "__main__":
    pipeline_config = config.PipelineConfig()
    pipeline_config.file.use_cached_original = False
    pipeline_config.file.use_cached_perturbed = True
    pipeline_config.file.current_mode = "val"
    pipeline_config.file.weighted = True
    pipeline_config.classify.analysis = False
    pipeline_config.classify.data_collection = False  #not pipeline_config.file.weighted

    # pipeline_config.classify.adaptive_weighting_per_head = 8.5

    main(pipeline_config)
