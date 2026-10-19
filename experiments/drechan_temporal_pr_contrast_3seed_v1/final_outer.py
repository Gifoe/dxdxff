"""No hidden continuation: fixed full-stage scoring only after verified FIT gate."""
from common import *

def main():
    assert jr(PUB/'internal/INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    assert jr(PUB/'internal/GATE.json')['continue_final'],'Internal gate failed: no new Outer'
    import assess
    assess.main('final')

if __name__=='__main__':main()
