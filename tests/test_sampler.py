from src.video.sampler import FrameSampler

def test_sampler_for_30fps_to_2fps():
    sampler = FrameSampler(30.0, 2.0)
    selected = [i for i in range(60) if sampler.should_sample(i)]
    assert selected == [0, 15, 30, 45]
