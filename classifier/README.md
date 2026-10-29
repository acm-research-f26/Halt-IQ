# classifier/

Placeholder for Sri's learned stopping classifier (from the `sri` branch, `implementation-3/`).

That classifier is a 6-parameter linear stop head over five per-round features
(`distance`, `round_fraction`, `distance_delta`, `critic_approved`, `has_previous_draft`),
trained by policy gradient on cached 6-round HotpotQA writer/critic trajectories.
Checkpoints are `study150/results/seed{0..4}/stop_head.pt` on that branch.
