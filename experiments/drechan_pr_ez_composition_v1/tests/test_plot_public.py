"""Presentation QA uses clearly watermarked synthetic aggregates only."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import plot_public

class PublicFigureTests(unittest.TestCase):
    def test_four_panels_render_synthetic_only(self):
        def estimate(**tags):return dict(scope='ALL',coordinate='ALL12',mean='0.2',ci_low='-0.1',ci_high='0.4',family_low='-0.2',family_high='0.5',**tags)
        t={'E1_GAIN':[],'PRIMARY':[],'E2_LOSS':[],'E2_ANOMALY':[],'SECONDARY':[]}
        for a in ['GLOBAL','SOURCE','CENTER','SCALE','FULL']:
            for group in ['EZ','NEZ']:t['E1_GAIN'].append(estimate(arm=a,group=group))
            for f in [.25,.5,.75,1.]:t['E2_LOSS'].append(estimate(arm=a,group='ALL',fraction=str(f)))
        for name in ['E1_H_FULL','E2_D_FULL_25_MINUS_100']:t['PRIMARY'].append(estimate(endpoint=name))
        for f in [.25,.5,.75,1.]:t['E2_ANOMALY'].append(estimate(endpoint='gap',fraction=str(f)))
        for f in [25,50,75]:t['SECONDARY'].append(estimate(endpoint=f'COMPOSITION_SLOPE_{f}'))
        import pymupdf
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='PRCOMP_SYNTHETIC_QA_') as tmp:
            p=Path(tmp);plot_public.render(t,p,synthetic=True)
            pdf=pymupdf.open(p/'pr_ez_composition.pdf');text=pdf[0].get_text();self.assertIn('SYNTHETIC QA',text)
            for word in ['EZ / NEZ','Reference-size sensitivity','Residual EZ association','Within-query composition']:self.assertIn(word,text)
            for f in pdf[0].get_fonts():self.assertGreater(len(pdf.extract_font(f[0])[3]),0)
            image=Image.open(p/'pr_ez_composition.png');image.load();self.assertGreater(image.width,7000);self.assertGreater(image.height,5000)
            self.assertIn('<text',(p/'pr_ez_composition.svg').read_text(encoding='utf-8'));pdf.close();image.close()

if __name__=='__main__':unittest.main()
