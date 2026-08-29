from pathlib import Path


def _read_text(relative_path):
    return (Path(__file__).resolve().parents[2] / relative_path).read_text(encoding="utf-8")


def test_adam_multiplier_hooks_present():
    mvau_text = _read_text("src/finn/custom_op/fpgadataflow/hls/matrixvectoractivation_hls.py")
    vvau_text = _read_text("src/finn/custom_op/fpgadataflow/hls/vectorvectoractivation_hls.py")
    adam_text = _read_text("custom_hls/adammultiplier.hpp")

    assert 'my_attrs["useAdamMultiplier"] = ("i", False, 0, {0, 1})' in mvau_text
    assert 'my_attrs["useAdamMultiplier"] = ("i", False, 0, {0, 1})' in vvau_text

    assert '#include "adammultiplier.hpp"' in mvau_text
    assert '#include "adammultiplier.hpp"' in vvau_text
    assert mvau_text.index('#include "adammultiplier.hpp"') < mvau_text.index('#include "mvau.hpp"')
    assert vvau_text.index('#include "adammultiplier.hpp"') < vvau_text.index('#include "mvau.hpp"')

    assert 'ap_resource_adam' in adam_text
    assert 'return adam_multiplier(c, d);' in adam_text
    assert 'using adam_hls::adam_multiplier;' in adam_text

    assert 'ap_resource_adam()' in mvau_text
    assert 'ap_resource_adam()' in vvau_text
