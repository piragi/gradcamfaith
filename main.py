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

    # How selective for S_f? (Top 20%, 10%, 5%)
    # Rationale: Your analysis showed S_f varies a lot. We need to test how much signal we need.
    # 80 is a good baseline, 95 is for extreme purity.
    percentile_thresholds = [80, 90]

    # How "stealthy"? (Bottom 30%, 50%)
    # Rationale: Test the definition of the "blind spot". Is it only the most ignored features or a wider set?
    attention_thresholds = [30, 50]

    # How many features to use? (CRITICAL PARAMETER)
    # Rationale: This is your noise vs. signal control. A small 'k' is high-purity. A large 'k' assumes many features work together.
    top_k_features_list = [25, 50, 75]

    # How strong to make the effect?
    # Rationale: Test gentle, moderate, and aggressive boosting. Too much can hurt faithfulness.
    base_strengths = [3.5, 5.0]

    for pct in percentile_thresholds:
        for attn in attention_thresholds:
            for k in top_k_features_list:
                for strength in base_strengths:
                    pipeline_config.classify.percentile_threshold = pct
                    pipeline_config.classify.attention_threshold = attn
                    pipeline_config.classify.top_k_features = k
                    pipeline_config.classify.base_strength = strength
                    print(
                        f"Config: features {pipeline_config.classify.percentile_threshold}%, attention {pipeline_config.classify.attention_threshold}%, top-k {pipeline_config.classify.top_k_features}, base_strength {pipeline_config.classify.base_strength}"
                    )
                    main(pipeline_config)
