"""One stage-specific frozen raw development and outer pass, no new training."""
from audit_sources import arguments
from evaluation import run
if __name__=='__main__':run(arguments(),raw=True)
