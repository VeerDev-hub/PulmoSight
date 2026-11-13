"""PulmoSight: TB vs Normal chest X-ray classification.

    pulmosight.models     Simple CNN, PulmoNet and EfficientNet-B0 definitions (shared with the notebooks)
    pulmosight.inference  Screener: preprocessing, calibrated prediction and the ensemble; also a CLI:
                          python -m pulmosight.inference path/to/xray.png --model ensemble
"""
