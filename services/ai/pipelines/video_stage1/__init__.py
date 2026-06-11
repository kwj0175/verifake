"""Video Stage 1 pipeline package."""


def run_video_stage1_preprocess(*args, **kwargs):
    from services.ai.pipelines.video_stage1.preprocess import (
        run_video_stage1_preprocess as _run_video_stage1_preprocess,
    )

    return _run_video_stage1_preprocess(*args, **kwargs)

def run_video_stage1_result_explainer(*args, **kwargs):
    from services.ai.pipelines.video_stage1.result_explainer import (
        run_video_stage1_result_explainer as _run_video_stage1_result_explainer,
    )

    return _run_video_stage1_result_explainer(*args, **kwargs)

__all__ = ["run_video_stage1_preprocess", "run_video_stage1_result_explainer"]
