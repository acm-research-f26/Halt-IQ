import unittest
from pc_strands.retention import derive_retained_episode


def panel(probabilities, drafts):
    n=len(probabilities)
    ep={"task_id":"q","policy":"strands","status":"completed","rounds":n,
        "decision_calls":n,"writer_input_tokens":321,"observed_prefix_model_seconds":4.2,
        "final_answer":drafts[n-1],"exact_match":False,"token_f1":0.0,
        "decisions":[{"round":i+1,"score":{"p_true":p}} for i,p in enumerate(probabilities)]}
    tr={"task_id":"q","rounds":[{"round":i+1,"draft":d,"status":"completed"}for i,d in enumerate(drafts)]}
    return ep,tr


class RetentionTests(unittest.TestCase):
    def test_earlier_candidate_can_be_kept_without_rewinding_cost(self):
        ep,tr=panel([.7,.4,.6],["right","wrong","wrong"])
        result=derive_retained_episode(ep,tr,["right"])
        self.assertEqual(result["selected_draft_round"],1)
        self.assertTrue(result["exact_match"])
        for key in ("rounds","decision_calls","writer_input_tokens","observed_prefix_model_seconds"):
            self.assertEqual(result[key],ep[key])
        self.assertEqual(ep["final_answer"],"wrong")

    def test_references_never_change_selection(self):
        ep,tr=panel([.4,.6],["right","wrong"])
        a=derive_retained_episode(ep,tr,["right"])
        b=derive_retained_episode(ep,tr,["wrong"])
        self.assertEqual(a["selected_draft_round"],b["selected_draft_round"])
        self.assertFalse(a["exact_match"])
        self.assertTrue(b["exact_match"])

    def test_future_draft_and_score_cannot_select_fallback(self):
        ep,tr=panel([.6,.5],["earlier","last","future"])
        ep["decisions"].append({"round":3,"score":{"p_true":1.0}})
        result=derive_retained_episode(ep,tr,["future"])
        self.assertEqual(result["selected_draft_round"],1)
        self.assertFalse(result["exact_match"])
        self.assertEqual(len(result["decisions"]),2)

    def test_ties_choose_earliest_and_failures_remain_zero(self):
        ep,tr=panel([.6,.6],["first","last"])
        self.assertEqual(derive_retained_episode(ep,tr,["first"])["selected_draft_round"],1)
        ep.update(status="failed",exact_match=True,token_f1=1)
        failed=derive_retained_episode(ep,tr,["first"])
        self.assertFalse(failed["exact_match"])
        self.assertEqual(failed["token_f1"],0)
        self.assertIsNone(failed["selected_draft_round"])

    def test_missing_or_invalid_probability_fails_closed(self):
        for probability in (True,float("nan"),-0.1,1.1):
            ep,tr=panel([probability],["answer"])
            with self.assertRaises(ValueError):derive_retained_episode(ep,tr,["answer"])
        ep,tr=panel([.5,.6],["first","last"])
        ep["decisions"].pop()
        with self.assertRaises(ValueError):derive_retained_episode(ep,tr,["first"])


if __name__=="__main__":unittest.main()
