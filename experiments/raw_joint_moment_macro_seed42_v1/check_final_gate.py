"""Verify real unfinished-internal guard denies final entry without data access."""
import final_stage as f
c=f.c


def main():
    terminal=c.PUB/'INTERNAL_TERMINAL.json';before=f.FINAL/'INPUT_FREEZE_PRIVATE.json'
    assert not terminal.exists(),'This negative gate check belongs before internal completion.'
    assert not before.exists(),'Do not modify an already initialized final stage.'
    try:f.setup()
    except FileNotFoundError as error:assert c.Path(error.filename)==terminal
    else:raise AssertionError('Final entry incorrectly allowed without internal terminal.')
    assert not before.exists()
    print('JOINT_RAW_FINAL_GATE_DENIED_BEFORE_INTERNAL_PASS',flush=True)


if __name__=='__main__':main()
