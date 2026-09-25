from run_neuroez_c import build_parser


def test_cli_exposes_explicit_lcbo_loss_mode_and_teacher_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(["--lcbo_loss_mode", "symmetric_lcbo", "--teacher_mode", "physiology_only"])
    assert args.lcbo_loss_mode == "symmetric_lcbo"
    assert args.teacher_mode == "physiology_only"

