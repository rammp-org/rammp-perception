from rammp_boxer.sync import FrameCache


def test_exact_and_nearest():
    c = FrameCache(size=5)
    for t in (100, 200, 300):
        c.put(t, f"m{t}")
    assert c.nearest(200, 0) == "m200"
    assert c.nearest(205, 0) is None
    assert c.nearest(205, 10) == "m200"
    assert c.nearest(260, 10) is None


def test_keeps_only_newest():
    c = FrameCache(size=2)
    for t in (1, 2, 3):
        c.put(t, t)
    assert c.nearest(1, 0) is None
    assert c.nearest(3, 0) == 3


def test_empty_and_clear():
    c = FrameCache()
    assert c.nearest(1, 10**9) is None
    c.put(1, "a")
    c.clear()
    assert c.nearest(1, 0) is None
