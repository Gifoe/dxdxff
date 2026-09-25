from dataclasses import dataclass
from pathlib import Path
@dataclass(frozen=True)
class ViewConfig:
 n_views:int=4;max_seizures:int=2;max_ez:int=24;max_nez:int=40;max_windows_per_phase:int=4
@dataclass(frozen=True)
class CachePaths:
 root:Path
 @property
 def patients(self):return self.root/'patients'
 @property
 def signature(self):return self.root/'cache_signature.json'
