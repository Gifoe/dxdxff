import importlib.util
from pathlib import Path


def _runner():
    path = Path(__file__).resolve().parents[2] / "scripts" / "task2" / "run_p2_q10_npam.py"
    spec = importlib.util.spec_from_file_location("npam_runner_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_cli_has_only_quick_or_outer_cv_and_no_selection_arguments():
    parser = _runner().build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--inner_folds" not in options and "--patience" not in options
    protocol = next(action for action in parser._actions if "--protocol" in action.option_strings)
    assert tuple(protocol.choices) == ("quick", "outer_cv")
