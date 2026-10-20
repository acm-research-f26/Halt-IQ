import copy
import unittest
from unittest.mock import patch
import torch
import hotpot as h

class Tests(unittest.TestCase):
    def setUp(self):
        self.rounds=[{'distance':None if i==0 else .02,'critic_approved':False,'quality':.5,'tokens':10,'answer_em':0,'answer':'x'} for i in range(6)]
    def test_quality(self):
        self.assertEqual(h.quality('The Eiffel Tower.','Eiffel Tower'),(1.,1.))
        self.assertEqual(h.quality('yes','no'),(0.,0.))
        self.assertEqual(h.quality('', ''),(1.,0.))
        self.assertAlmostEqual(h.quality('red blue','red')[1],2/3)
    def test_no_reward_leakage(self):
        changed=copy.deepcopy(self.rounds)
        for r in changed:r.update(quality=1,answer_em=1,answer='reference answer')
        for i in range(6):torch.testing.assert_close(h.features(self.rounds,i),h.features(changed,i))
    def test_hard_cap(self):
        model=h.StopHead()
        with torch.no_grad():model.head.weight.zero_();model.head.bias.fill_(-100)
        self.assertEqual(h.rollout(model,self.rounds,True)[0],5)
    def test_stop_first(self):
        model=h.StopHead()
        with torch.no_grad():model.head.weight.zero_();model.head.bias.fill_(100)
        self.assertEqual(h.rollout(model,self.rounds,True)[0],0)
    def test_distance_patience(self):
        self.assertEqual(h.choose('distance',None,self.rounds),2)
    def test_reward_cost(self):
        self.assertGreater(h.reward(self.rounds,0,.1),h.reward(self.rounds,5,.1))
    def test_test_partition_preserved(self):
        self.assertEqual(h.split_for({'id':'q','split':'test'}),'test')
    def test_policy_gradient(self):
        model=h.StopHead();loss=-torch.distributions.Bernoulli(logits=model(h.features(self.rounds,0))).log_prob(torch.tensor(1.))
        loss.backward();self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()),0)

if __name__=='__main__':unittest.main()
