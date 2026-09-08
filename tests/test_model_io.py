from tennetsac.utils import model_io


def test_legacy_ten_checkpoint_loader_is_not_part_of_model_io() -> None:
    assert not hasattr(model_io, "load_all_Gamma_models")
