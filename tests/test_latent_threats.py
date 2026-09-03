import torch

from model.modules import ComposedFeatureTransformer
from model.modules.features import (
    LatentThreats,
    get_available_features,
    get_feature_cls,
)
from model.quantize import QuantizationConfig, QuantizationManager


def test_latent_threats_attributes():
    assert LatentThreats.NUM_INPUTS == 12288
    assert LatentThreats.NUM_REAL_FEATURES == 12288
    assert LatentThreats.MAX_ACTIVE_FEATURES == 8
    assert LatentThreats.EXPORT_WEIGHT_DTYPE == torch.int8
    assert LatentThreats.FEATURE_NAME == "LatentThreats"
    assert LatentThreats.INPUT_FEATURE_NAME == "LatentThreats"


def test_latent_threats_init_and_weights():
    num_outputs = 1024
    num_psqt = 8
    feature = LatentThreats(num_outputs)

    # Check weight dimensions
    assert feature.weight.shape == (12288, num_outputs)
    assert feature.merged_weight().shape == (12288, num_outputs)

    # Initialize weights (PSQT columns must be zeroed)
    feature.init_weights(num_psqt_buckets=num_psqt, nnue2score=600.0)
    l1 = num_outputs - num_psqt
    assert torch.all(feature.weight[:, l1:] == 0.0)

    # Test weight clipping
    q_cfg = QuantizationConfig()
    qm = QuantizationManager(q_cfg)
    feature.weight.data.fill_(10.0)
    feature.clip_weights(qm)
    assert torch.all(feature.weight <= qm.max_threat_weight + 1e-6)
    assert torch.all(feature.weight >= qm.min_threat_weight - 1e-6)


def test_latent_threats_factory_registration():
    available = get_available_features()
    assert "LatentThreats" in available
    assert "Latent_Threats" in available

    cls = get_feature_cls("LatentThreats")
    assert cls == [LatentThreats]


def test_composed_with_latent_threats():
    feature_name = "Full_Threats+PP_3Wide+HalfKAv2_hm^+LatentThreats"
    classes = get_feature_cls(feature_name)

    q_cfg = QuantizationConfig()
    qm = QuantizationManager(q_cfg)
    cft = ComposedFeatureTransformer(classes, 1024, 8, qm)

    # Total real inputs should be 86896 + 12288 = 99184
    assert cft.NUM_REAL_FEATURES == 99184
    # Max active should be 128 (FullThreats) + 128 (PP_3Wide) + 32 (HalfKAv2) + 8 (LatentThreats) = 296
    assert cft.MAX_ACTIVE_FEATURES == 296
    assert cft.FEATURE_NAME == feature_name
