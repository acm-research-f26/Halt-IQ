import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import study, hotpot
class StudyTests(unittest.TestCase):
 def test_question_partition_integrity(self):
  rows=json.loads((study.STUDY/'questions.json').read_text());old=json.loads((study.ROOT/'data/scenarios_hotpot.json').read_text())
  self.assertEqual(len(rows),600);self.assertEqual(len({r['id'] for r in rows}),600)
  self.assertFalse({r['id'] for r in rows}&{r['id'] for r in old})
  self.assertEqual([sum(r['split']==s for r in rows) for s in ['train','dev','test']],[300,100,200])
  self.assertTrue(all(len(r['contexts'])==10 for r in rows))
 def test_oracle_and_cost_accounting(self):
  rr=[{'answer':'x','quality':v,'answer_em':0,'tokens':15,'distance':None if i==0 else .1,'critic_approved':False,
   'writer':{'usage':{'prompt_tokens':5,'completion_tokens':3,'total_tokens':8},'seconds':2},
   'critic':{'usage':{'prompt_tokens':6,'completion_tokens':1,'total_tokens':7},'seconds':1}} for i,v in enumerate([.1,.9,.9,.1,.1,.1])]
  item={'id':'hotpot_x','question':'q','rounds':rr}
  rows=study.rows_for([item],hotpot.StopHead(),0);oracle=next(r for r in rows if r['policy']=='oracle_f1')
  self.assertEqual(oracle['rounds'],2);self.assertEqual(oracle['total_tokens'],30)
  self.assertEqual(oracle['prompt_tokens']+oracle['completion_tokens'],30)
  self.assertEqual(oracle['writer_tokens']+oracle['critic_tokens'],30)
  self.assertEqual(oracle['llm_calls'],4);self.assertEqual(oracle['request_seconds'],6)
 def test_no_gold_in_prompts(self):
  # Run full generation against a fake transport, verifying actual constructed inputs.
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);q=[{'id':'hotpot_fake','question':'q','contexts':['evidence'],'ground_truth':'SECRET_GOLD','split':'train','type':'bridge'}]
   study.save(root/'questions.json',q);study.save(root/'protocol.json',{'dataset_sha256':study.digest(root/'questions.json')})
   captured=[]
   def fake(endpoint,messages,*rest):
    captured.append(messages);return {'text':'answer','usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3},'seconds':.1}
   class Response:
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def read(self):return b'{}'
   with patch.object(study,'STUDY',root),patch.object(study,'request',fake),patch.object(study.urllib.request,'urlopen',return_value=Response()):study.generate('http://localhost/v1/chat/completions',1)
   self.assertEqual(len(captured),12);self.assertNotIn('SECRET_GOLD',json.dumps(captured))
   result=json.loads((root/'cache/hotpot_fake.json').read_text());self.assertEqual(len(result['rounds']),6)
if __name__=='__main__':unittest.main()
