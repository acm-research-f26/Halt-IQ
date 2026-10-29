import importlib.util,json,tempfile,unittest
from pathlib import Path
import torch
from classifier.model import features,StopHead,StopPredictor,FEATURES
ROOT=Path(__file__).resolve().parent
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items=json.loads((ROOT/'inputs/trajectories.json').read_text())
        spec=importlib.util.spec_from_file_location('old',ROOT/'reference/original_hotpot.py')
        cls.old=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.old)
    def test_original_feature_parity_all_900_rounds(self):
        for item in self.items:
            rr=item['rounds']
            for i in range(6):self.assertTrue(torch.equal(features(rr[:i+1],'original'),self.old.features(rr,i)))
    def test_original_inference_parity_all_seeds_and_rounds(self):
        for seed in range(5):
            p=StopPredictor(ROOT/f'reference/original_seed{seed}.pt')
            m=self.old.StopHead();m.load_state_dict(torch.load(ROOT/f'reference/original_seed{seed}.pt',weights_only=True)['weights'])
            for item in self.items:
                for i in range(6):
                    with torch.no_grad():expected=torch.sigmoid(m(self.old.features(item['rounds'],i))).item()
                    self.assertEqual(p.predict(item['question'],item['rounds'][:i+1]),expected)
    def test_no_labels_costs_or_question_used(self):
        rr=self.items[0]['rounds'][:3]
        clean=[{k:r[k] for k in ['answer','critic_approved','strands']} for r in rr]
        for variant in FEATURES:
            self.assertTrue(torch.equal(features(rr,variant),features(clean,variant)))
        p=StopPredictor(ROOT/'reference/original_seed0.pt')
        self.assertEqual(p.predict('anything',rr),p.predict('different',clean))
    def test_prefix_denominator_is_cap_not_prefix_length(self):
        rr=self.items[0]['rounds']
        self.assertAlmostEqual(features(rr[:1],'original')[1].item(),1/6)
        self.assertEqual(features(rr[:3],'original')[1].item(),.5)
    def test_variants_change_only_intended_inputs(self):
        rr=self.items[0]['rounds'][:3];a=features(rr,'original');b=features(rr,'replace');c=features(rr,'add')
        self.assertTrue(torch.equal(c[:5],a));self.assertAlmostEqual(c[5].item(),rr[-1]['strands'])
        self.assertTrue(torch.equal(a[[0,1,2,4]],b[[0,1,2,4]]));self.assertAlmostEqual(b[3].item(),rr[-1]['strands'])
        self.assertEqual(sum(p.numel() for p in StopHead('add').parameters()),7)
    def test_invalid_or_missing_inputs_rejected(self):
        for score in [None,float('nan'),float('inf'),-1,1.1,True]:
            with self.assertRaises(ValueError):features([{'answer':'a','strands':score}],'replace')
        with self.assertRaises(ValueError):features([{'answer':'a'}],'original')
        with self.assertRaises(ValueError):features([{'round':2,'answer':'a','critic_approved':True}],'original')
    def test_replacement_does_not_require_old_critic(self):
        self.assertEqual(len(features([{'answer':'a','strands':.7}],'replace')),5)
    def test_round_cap_is_enforced_even_for_low_probability(self):
        with tempfile.TemporaryDirectory() as d:
            m=StopHead();m.head.weight.data.zero_();m.head.bias.data.fill_(-100)
            pth=Path(d)/'c.pt';torch.save({'weights':m.state_dict(),'features':FEATURES['original']},pth)
            p=StopPredictor(pth);rr=self.items[0]['rounds']
            self.assertFalse(p.should_stop('',rr[:5]));self.assertTrue(p.should_stop('',rr))
if __name__=='__main__':unittest.main()
