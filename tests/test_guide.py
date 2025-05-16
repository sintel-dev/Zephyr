from zephyr_ml.core import GuideHandler, guide


class DummyObject:
    def __init__(self):
        producers_and_getters = [
            ([self.step0_key, self.step0_set], [self.step0_getter]),
            ([self.step1_key, self.step1_set], [self.step1_getter]),
            ([self.step2_key, self.step2_set], [self.step2_getter])
        ]
        set_methods = {
            self.step0_set.__name__,
            self.step1_set.__name__,
            self.step2_set.__name__
        }
        self._guide_handler = GuideHandler(producers_and_getters, set_methods)

    @guide
    def step0_key(self):
        return "step0_key_result"

    @guide
    def step0_set(self):
        return "step0_set_result"

    @guide
    def step0_getter(self):
        return "step0_get_result"

    @guide
    def step1_key(self):
        return "step1_key_result"

    @guide
    def step1_set(self):
        return "step1_set_result"

    @guide
    def step1_getter(self):
        return "step1_get_result"

    @guide
    def step2_key(self):
        return "step2_key_result"

    @guide
    def step2_set(self):
        return "step2_set_result"

    @guide
    def step2_getter(self):
        return "step2_get_result"


def test_forward_key_steps():
    """Test performing key steps in forward order"""
    obj = DummyObject()

    # First step should work without warnings
    assert obj.step0_key() == "step0_key_result"

    # Second step should work without warnings since previous step is up to
    # date
    assert obj.step1_key() == "step1_key_result"

    # Third step should work without warnings since previous step is up to date
    assert obj.step2_key() == "step2_key_result"


def test_set_methods_can_skip(caplog):
    """Test that set methods can skip steps"""
    obj = DummyObject()

    # Set methods should work in any order and start new iterations
    assert obj.step2_set() == "step2_set_result"  # Skip to step 2
    assert "[GUIDE] STALE WARNING" in caplog.text

    assert obj.step0_set() == "step0_set_result"  # Go back to step 0
    assert "[GUIDE] STALE WARNING" in caplog.text
    assert obj.step1_set() == "step1_set_result"  # Do step 1
    assert "[GUIDE] STALE WARNING" in caplog.text


def test_key_methods_require_previous_step(caplog):
    """Test that key methods require the previous step to be up to date"""
    obj = DummyObject()

    # Try to do step 1 without doing step 0 first
    obj.step1_key()
    assert "[GUIDE] INCONSISTENCY WARNING" in caplog.text

    # Do step 0, then step 1 should work
    obj.step0_key()
    assert obj.step1_key() == "step1_key_result"


def test_stale_data_warning(caplog):
    """Test warning when data becomes stale"""
    obj = DummyObject()

    # Complete steps 0 and 1
    obj.step0_key()
    obj.step1_key()

    # Go back to step 0 with set method (allowed, but warns about stale data)
    obj.step0_set()
    assert "[GUIDE] STALE WARNING" in caplog.text


def test_getter_with_stale_data(caplog):
    """Test getting data that may be stale"""
    obj = DummyObject()

    # Complete steps 0 and 1
    obj.step0_key()
    obj.step1_key()

    # Go back to step 0 with set method
    obj.step0_set()

    # Try to get data from step 1, should warn about stale data
    obj.step1_getter()
    assert "[GUIDE] STALE WARNING" in caplog.text


def test_getter_with_missing_key(caplog):
    """Test getting data when the key method hasn't been run"""
    obj = DummyObject()

    # Try to get data without running key method first
    obj.step1_getter()
    assert "[GUIDE] INCONSISTENCY WARNING" in caplog.text


def test_key_method_after_stale_data(caplog):
    """Test that key methods cannot be run when previous step is stale"""
    obj = DummyObject()

    # Complete steps 0 and 1
    obj.step0_key()
    obj.step1_key()

    # Go back to step 0 with set method
    obj.step1_set()
    obj.step1_key()
    assert "[GUIDE] INCONSISTENCY WARNING" in caplog.text


def test_multiple_iterations():
    """Test multiple iterations through the steps"""
    obj = DummyObject()

    # First iteration with key methods
    assert obj.step0_key() == "step0_key_result"
    assert obj.step1_key() == "step1_key_result"

    # Second iteration starting with set method
    assert obj.step0_set() == "step0_set_result"
    # Can't do step 1 with key method after set without redoing step 0 key
    assert obj.step1_key() == "step1_key_result"


def test_guide_decorator():
    """Test that the guide decorator properly wraps methods"""
    obj = DummyObject()

    # Check that the decorator preserves function metadata
    assert obj.step0_key.__name__ == "step0_key"
    assert obj.step0_getter.__name__ == "step0_getter"

    # Check that the decorator routes through the guide handler
    assert obj.step0_key() == "step0_key_result"
    assert obj.step0_getter() == "step0_get_result"
