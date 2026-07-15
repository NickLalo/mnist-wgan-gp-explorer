from mnist_wgan.paths import BUNDLED_CHECKPOINT, LOCAL_CHECKPOINT, default_checkpoint_path


def test_checkpoint_fallback_prefers_a_local_training_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    BUNDLED_CHECKPOINT.parent.mkdir(parents=True)
    BUNDLED_CHECKPOINT.touch()
    assert default_checkpoint_path() == BUNDLED_CHECKPOINT

    LOCAL_CHECKPOINT.parent.mkdir(parents=True)
    LOCAL_CHECKPOINT.touch()
    assert default_checkpoint_path() == LOCAL_CHECKPOINT
